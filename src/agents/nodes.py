import ast
import json
import multiprocessing
import os
import re
import sys
import time
import asyncio
import contextlib
from typing import Any, Dict
import logfire
import numpy as np
import pandas as pd

from src.agents.crews import run_de_crew_instrumented, run_mle_crew_instrumented
from src.agents.tools import lookup_playbook, upsert_playbook
from src.domain.models import OrchestratorState
from src.observability_bus import publish_stage

_MLE_OUTPUT_BUFFER: str = ""

# A challenger must beat the current champion strictly to be promoted.
# RMSE-style metrics are lower-is-better; F1/silhouette/ROC-AUC are higher-is-better.
LOWER_IS_BETTER_TASKS = {"REGRESSION"}

# --- DRY UTILITY: Universal Path Resolver & Code Extractor ---


def describe_schema(df: pd.DataFrame) -> str:
    """Compact dtype and null summary, so the model sees the data without spending a request on it."""
    return ", ".join(f"{c}: {df[c].dtype} ({int(df[c].isnull().sum())} nulls)" for c in df.columns)


def resolve_data_path(state: dict) -> str:
    """Forcefully resolves dataset path based on trace_id or explicit payload_uri/data_path."""
    trace_id = str(state.get("trace_id", "")).lower()

    # Priority 1: Trace ID heuristic override (fixes default state pollution)
    if "ecom" in trace_id:
        return "data/ecommerce_sales.parquet"
    if "chaos" in trace_id:
        return "data/chaos_ingestion.parquet"

    # Priority 2: Explicit path if non-default
    path = state.get("data_path") or state.get("payload_uri")
    if path and path != "data/default_ingestion.parquet":
        return path

    return "data/default_ingestion.parquet"


def extract_and_sanitize_code(raw_llm_output: str) -> str:
    """Aggressively extracts Python code from LLM outputs, bypassing fragile regex, while keeping AST validation."""
    if not raw_llm_output:
        return "# No code generated"

    cleaned_code = raw_llm_output

    # 1. Bruteforce strip the starting markdown wrapper
    if "```python" in cleaned_code:
        cleaned_code = cleaned_code.split("```python", 1)[1]
    elif "```" in cleaned_code:
        cleaned_code = cleaned_code.split("```", 1)[1]

    # 2. Bruteforce strip the trailing wrapper
    if "```" in cleaned_code:
        cleaned_code = cleaned_code.split("```")[0]

    code_str = cleaned_code.strip()

    if not code_str:
        return "# No valid code block found in LLM output"

    try:
        ast.parse(code_str)
        return code_str
    except SyntaxError:
        pass

    # Trim trailing lines if syntax error occurs
    lines = code_str.splitlines()
    while len(lines) > 5:
        lines.pop()
        candidate = "\n".join(lines).strip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return code_str  # Return raw candidate anyway so the dashboard shows what the LLM wrote


# ================================================================================
# NODE 1 & 2: DATA ENGINEERING LOOP
# ================================================================================


async def run_de_crew_node(state: OrchestratorState) -> Dict[str, Any]:
    trace_id = state.get("trace_id", "unknown")
    retries = state.get("de_retry_count", 0)

    escalated_from_mle = state.get("status") == "MLE_FAILED"
    if escalated_from_mle:
        retries = 0

    loop_msg = f" (Self-Healing Retry {retries}/3)" if retries > 0 else ""
    if escalated_from_mle:
        loop_msg = " (Escalated from MLE - re-cleaning data)"
    await publish_stage(trace_id, "de_crew", "RUNNING", detail=f"Deploying Janitor Crew{loop_msg}...")

    de_state = dict(state)
    if escalated_from_mle:
        de_state["de_retry_count"] = 0

    result = await _run_de_crew_node_impl(de_state)
    
    # 🛡️ THE FIX: Safely reset MLE retries so the Scientist gets a fresh try
    if escalated_from_mle:
        result["mle_retry_count"] = 0
        
    ok = result.get("status") == "DE_CODE_GENERATED"

    generated_code = result.get("final_patch", "# No code generated")
    bt = chr(96) * 3
    success_detail = f"**🧹 Janitor Crew generated the following Python patch:**\n\n{bt}python\n{generated_code}\n{bt}"
    error_detail = result.get("de_error_feedback") or "Failed to generate patch"

    await publish_stage(
        trace_id, "de_crew", "SUCCESS" if ok else "FAILED",
        detail=success_detail if ok else error_detail,
    )
    return result


async def _run_de_crew_node_impl(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    retries = state.get("de_retry_count", 0)
    print(
        f"\n{'='*80}\n [NODE 1/5] STARTING DATA ENGINEERING CREW (RETRY {retries}/3)\n{'='*80}\n", flush=True)

    trace_id = state.get("trace_id", "unknown")
    data_path = resolve_data_path(state)
    err_fb = state.get("de_error_feedback")
    prev_code = state.get("de_previous_code")

    actual_columns = []
    schema_profile = "unknown"
    if os.path.exists(data_path):
        try:
            df_sample = pd.read_parquet(data_path)
            actual_columns = df_sample.columns.tolist()
            schema_profile = describe_schema(df_sample)
        except Exception:
            pass

    # One embedding call instead of an agent tool loop (each tool turn cost a model request).
    signature = f"{err_fb or 'schema drift and data quality degradation'} | {schema_profile}"
    playbook_context = await asyncio.to_thread(lookup_playbook, signature)

    inputs = {
        "trace_id": trace_id,
        "schema_columns": str(actual_columns),
        "schema_profile": schema_profile,
        "playbook_context": playbook_context.strip(),
        "data_path": data_path
    }

    try:
        raw_output = await run_de_crew_instrumented(inputs, err_fb, prev_code)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(
            f"[Agent: DE Crew] 🚨 Fatal LLM Execution Error: {error_msg}", flush=True)

        if "429" in error_msg or "Resource exhausted" in error_msg:
            print(
                "[Agent: DE Crew] ⏳ Rate limit hit. Cooling down for 30s...", flush=True)
            await asyncio.sleep(30)

        return {
            "status": "DE_FAILED",
            "de_error_feedback": f"API Provider Error: {error_msg}",
            "de_previous_code": prev_code,
            "de_retry_count": retries + 1
        }

    patch = extract_and_sanitize_code(raw_output)
    if not patch or patch.startswith("# No valid"):
        return {
            "status": "DE_FAILED",
            "de_error_feedback": "SyntaxError: No valid Python code block generated.",
            "de_previous_code": raw_output,
            "de_retry_count": retries + 1
        }

    return {"final_patch": patch, "status": "DE_CODE_GENERATED", "de_retry_count": retries + 1, "data_path": data_path}


def _os_process_de_sandbox(code_string: str, data_path: str, queue: multiprocessing.Queue):
    # 🛡️ BULLETPROOF SAFETY STRIP: Clean it one last time before exec()
    code_string = extract_and_sanitize_code(code_string)

    if os.getenv("LOGFIRE_TOKEN"):
        logfire.configure(send_to_logfire="if-token-present")
    try:
        with logfire.span("sandbox.de_execution"):
            df = pd.read_parquet(data_path)

            if df.empty:
                queue.put({
                    "status": "DE_FAILED",
                    "error": f"AssertionError: Source file at '{data_path}' is already empty (0 rows)."
                })
                return

            original_row_count = len(df)
            original_columns = set(df.columns)

            for col in df.columns:
                if hasattr(df[col].dtype, "pyarrow_dtype") or "arrow" in str(df[col].dtype).lower():
                    df[col] = df[col].astype(object)

            unified_namespace = {
                "__builtins__": __builtins__,
                "df": df,
                "pd": pd,
                "np": np
            }
            exec(code_string, unified_namespace)

            healed_df = unified_namespace.get("df")

            if healed_df is None or not isinstance(healed_df, pd.DataFrame):
                queue.put({
                    "status": "DE_FAILED",
                    "error": "AssertionError: Patch did not leave a valid 'df' DataFrame in scope.",
                })
                return

            if healed_df.empty:
                queue.put({
                    "status": "DE_FAILED",
                    "error": f"AssertionError: DataFrame is empty (0 rows) after patch ran.",
                })
                return

            MIN_RETAINED_ROW_FRACTION = 0.5
            retained_fraction = len(healed_df) / original_row_count
            if retained_fraction < MIN_RETAINED_ROW_FRACTION:
                queue.put({
                    "status": "DE_FAILED",
                    "error": f"AssertionError: Patch dropped {100 * (1 - retained_fraction):.1f}% of rows.",
                })
                return

            missing_columns = original_columns - set(healed_df.columns)
            if missing_columns:
                queue.put({
                    "status": "DE_FAILED",
                    "error": f"AssertionError: Patch removed column(s) {sorted(missing_columns)}.",
                })
                return

            if healed_df.isnull().sum().sum() != 0:
                for col in healed_df.columns:
                    if pd.api.types.is_numeric_dtype(healed_df[col]):
                        healed_df[col] = healed_df[col].fillna(0)
                    else:
                        healed_df[col] = healed_df[col].fillna("UNKNOWN")

            backup_path = f"{data_path}.bak"
            try:
                df.to_parquet(backup_path, index=False)
            except Exception:
                pass

            tmp_path = f"{data_path}.tmp"
            healed_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, data_path)

            queue.put({"status": "SUCCESS"})

    except Exception as e:
        queue.put({"status": "DE_FAILED",
                  "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        try:
            logfire.force_flush()
        except Exception:
            pass


async def run_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    trace_id = state.get("trace_id", "unknown")
    await publish_stage(trace_id, "executor", "RUNNING")
    result = await _run_executor_node_impl(state)
    ok = result.get("status") == "SUCCESS"
    await publish_stage(
        trace_id, "executor", "SUCCESS" if ok else "FAILED",
        detail=result.get("de_error_feedback") or "Sandbox patch applied",
    )
    return result


async def _run_executor_node_impl(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    print(
        f"\n{'='*80}\n [NODE 2/5] STARTING EXECUTOR SANDBOX\n{'='*80}\n", flush=True)

    data_path = resolve_data_path(state)
    code_to_run = state.get("final_patch", "")

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_os_process_de_sandbox, args=(code_to_run, data_path, queue))
    process.start()
    process.join(timeout=15.0)

    if process.is_alive():
        process.kill()
        process.join()
        return {"status": "DE_FAILED", "de_error_feedback": "TimeoutError: Execution exceeded 15s.", "de_previous_code": code_to_run}

    if not queue.empty():
        res = queue.get()
        if res.get("status") == "SUCCESS":
            return {"status": "SUCCESS", "de_error_feedback": None, "de_previous_code": None, "data_path": data_path}
        return {"status": "DE_FAILED", "de_error_feedback": res.get("error"), "de_previous_code": code_to_run}
    return {"status": "DE_FAILED", "de_error_feedback": "Sandbox Crashed Silently.", "de_previous_code": code_to_run}

# ================================================================================
# NODE 3: MEMORY UPDATER (QDRANT)
# ================================================================================


async def run_memory_updater_node(state: OrchestratorState) -> Dict[str, Any]:
    trace_id = state.get("trace_id", "unknown")
    await publish_stage(trace_id, "memory_updater", "RUNNING")
    result = await _run_memory_updater_node_impl(state)
    ok = result.get("status") == "MEMORY_UPDATED"
    await publish_stage(
        trace_id, "memory_updater", "SUCCESS" if ok else "FAILED",
        detail="Playbook committed to Qdrant" if ok else "Qdrant commit failed",
    )
    return result


async def _run_memory_updater_node_impl(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    print(
        f"\n{'='*80}\n [NODE 3/5] MEMORY UPDATER (QDRANT)\n{'='*80}\n", flush=True)
    try:
        upsert_playbook(
            f"Anomaly trace {state.get('trace_id')}", state.get("final_patch", ""))
        return {"status": "MEMORY_UPDATED"}
    except Exception:
        return {"status": "MEMORY_UPDATE_FAILED"}

# ================================================================================
# NODE 4 & 5: MLE REGISTRY LOOP (TASK-AWARE AUTO-PROFILING)
# ================================================================================


async def run_mle_crew_node(state: OrchestratorState) -> Dict[str, Any]:
    trace_id = state.get("trace_id", "unknown")
    retries = state.get("mle_retry_count", 0)

    # 🚀 Announce the loop iteration to the dashboard!
    loop_msg = f" (Self-Healing Retry {retries}/3)" if retries > 0 else ""
    await publish_stage(trace_id, "mle_crew", "RUNNING", detail=f"Deploying Scientist Crew{loop_msg}...")

    result = await _run_mle_crew_node_impl(state)
    ok = result.get("status") == "MLE_CODE_GENERATED"

    # 🛠️ Extract the actual Python script from the result dictionary
    generated_code = result.get("mle_code", "# No code generated")

    # 🚀 Format it beautifully for the dashboard
    success_detail = f"**🧬 Scientist Crew generated the following Scikit-Learn script:**\n\n```python\n{generated_code}\n```"
    error_detail = result.get(
        "mle_error_feedback") or "Failed to generate training script"

    # 🌟 FIX: Updated stage identifier from 'model_engineering' to 'mle_crew' to match UI grid
    await publish_stage(
        trace_id, "mle_crew", "SUCCESS" if ok else "FAILED",
        detail=success_detail if ok else error_detail,
    )
    return result


async def _run_mle_crew_node_impl(state: OrchestratorState) -> Dict[str, Any]:
    global _MLE_OUTPUT_BUFFER
    sys.stdout.flush()
    retries = state.get("mle_retry_count", 0)
    data_path = resolve_data_path(state)

    print(
        f"\n{'='*80}\n [NODE 4/5] STARTING MLE CREW (RETRY {retries}/3)\n{'='*80}\n", flush=True)

    print(
        f"[Task Profiler] Successfully locked onto dataset: '{data_path}'", flush=True)

    task_type = "BINARY_CLASSIFICATION"
    target_col = None
    schema_profile = "unknown"

    if os.path.exists(data_path):
        try:
            df_temp = pd.read_parquet(data_path)
            schema_profile = describe_schema(df_temp)

            possible_targets = [c for c in ['is_fraud', 'target', 'label', 'category',
                                            'status_code', 'price', 'transaction_amount'] if c in df_temp.columns]

            if possible_targets:
                target_col = possible_targets[0]
            elif len(df_temp.columns) > 1 and df_temp.columns[-1] not in ['trace_id', 'timestamp', 'account_id']:
                target_col = df_temp.columns[-1]

            if not target_col or target_col not in df_temp.columns:
                task_type = "CLUSTERING"
                target_col = "NONE (UNSUPERVISED)"
            else:
                target_series = df_temp[target_col].dropna()
                unique_count = target_series.nunique()
                is_numeric = pd.api.types.is_numeric_dtype(target_series)

                if unique_count <= 1:
                    task_type = "CLUSTERING"
                elif unique_count == 2:
                    task_type = "BINARY_CLASSIFICATION"
                elif 2 < unique_count <= 20 or not is_numeric:
                    task_type = "MULTICLASS_CLASSIFICATION"
                else:
                    task_type = "REGRESSION"

            print(
                f"[Task Profiler] Auto-detected Task: '{task_type}' | Target Column: '{target_col}'", flush=True)
        except Exception as e:
            print(
                f"[Task Profiler] Schema profiling warning: {e}. Defaulting to BINARY_CLASSIFICATION.", flush=True)

    inputs = {
        "data_path": data_path,
        "task_type": task_type,
        "TASK_TYPE": task_type,
        "target_col": target_col,
        "TARGET_COL": target_col,
        "schema_profile": schema_profile,
        "trace_id": state.get("trace_id", "tx_unknown")
    }

    try:
        raw_output = await run_mle_crew_instrumented(
            inputs,
            state.get("mle_error_feedback"),
            state.get("mle_previous_code")
        )
        _MLE_OUTPUT_BUFFER = raw_output
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(
            f"[Agent: MLE Crew] 🚨 Fatal LLM Execution Error: {error_msg}", flush=True)

        if "429" in error_msg or "Resource exhausted" in error_msg:
            print(
                "[Agent: MLE Crew] ⏳ Rate limit hit again. Cooling down for 30s...", flush=True)
            await asyncio.sleep(30)

        return {
            "status": "MLE_FAILED",
            "mle_error_feedback": f"API Provider Error: LLM execution failed. {error_msg}",
            "mle_previous_code": state.get("mle_previous_code"),
            "mle_retry_count": retries + 1
        }

    code = extract_and_sanitize_code(raw_output)
    if not code or code.startswith("# No valid"):
        print("[Agent: MLE Crew] ⚠️ No valid python block generated.", flush=True)
        return {
            "status": "MLE_FAILED",
            "mle_error_feedback": "SyntaxError: No valid Python code block generated. Please wrap your code in triple backticks.",
            "mle_previous_code": raw_output,
            "mle_retry_count": retries + 1
        }

    return {"status": "MLE_CODE_GENERATED", "mle_output": code, "mle_retry_count": retries + 1, "data_path": data_path}


def _is_better(task_type: str, new_score: float, champion_score) -> bool:
    if champion_score is None:
        return True
    if task_type in LOWER_IS_BETTER_TASKS:
        return new_score < champion_score
    return new_score > champion_score


REGISTRY_PATH = "models/model_registry.json"
CANDIDATE_ONNX_PATH = "models/candidate_model.onnx"
CHAMPION_ONNX_PATH = "models/pipeline_model.onnx"

# Enterprise gates. A metric the sandbox can't measure fails its gate instead of
# defaulting to a passing value.
MAX_INFERENCE_LATENCY_MS = 5.0
MIN_FAIRNESS_SCORE = 0.8
FAIRNESS_ATTRIBUTE = os.getenv("PHOENIX_FAIRNESS_ATTRIBUTE", "category")
MIN_FAIRNESS_SLICE_ROWS = 30


def _load_registry(registry_path: str) -> dict:
    registry = {"active_champion": None, "history": []}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "history" in loaded:
                    registry = loaded
        except Exception:
            pass
    return registry


def _would_promote(task_type: str, primary_score: float, passed_gates: bool, registry: dict) -> bool:
    champion = registry.get("active_champion")
    champion_score = champion["metrics"].get("primary_score") if champion else None
    return passed_gates and _is_better(task_type, primary_score, champion_score)


def _update_model_registry(task_type: str, metrics: dict, primary_score: float,
                           candidate_onnx_path: str, champion_onnx_path: str,
                           registry_path: str, passed_gates: bool = True) -> dict:
    """Deterministic Champion/Challenger bookkeeping. Only the auditor calls this."""
    registry = _load_registry(registry_path)
    won = _would_promote(task_type, primary_score, passed_gates, registry)

    entry = {
        "version": f"v{len(registry['history']) + 1}_{int(time.time())}",
        "task_type": task_type,
        "metrics": {**metrics, "primary_score": primary_score},
        "status": "PRODUCTION" if won else "CHALLENGER (Failed Gates or Score)",
        "timestamp": time.time(),
    }

    if won:
        with open(candidate_onnx_path, "rb") as src, open(champion_onnx_path, "wb") as dst:
            dst.write(src.read())
        registry["active_champion"] = entry

    registry["history"].append(entry)
    registry["history"] = registry["history"][-50:]

    tmp_path = registry_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    os.replace(tmp_path, registry_path)

    return registry


def compute_fairness(X_val, y_val, y_pred, attribute: str = FAIRNESS_ATTRIBUTE,
                     min_rows: int = MIN_FAIRNESS_SLICE_ROWS):
    """Worst weighted F1 across slices of the protected attribute, or None when it can't be measured.

    Needs at least two slices of min_rows each; one-row slices (e.g. an ID column) made the
    score swing between 0.0 and 1.0 on a single prediction.
    """
    from sklearn.metrics import f1_score

    if X_val is None or y_val is None or y_pred is None:
        return None
    if not isinstance(X_val, pd.DataFrame) or attribute not in X_val.columns:
        return None

    groups = X_val[attribute].to_numpy()
    y_true = np.asarray(y_val)
    y_hat = np.asarray(y_pred)
    if not (len(groups) == len(y_true) == len(y_hat)):
        return None

    scores = []
    for value in pd.unique(groups):
        mask = groups == value
        if mask.sum() >= min_rows:
            scores.append(f1_score(y_true[mask], y_hat[mask], average="weighted", zero_division=0))
    return float(min(scores)) if len(scores) >= 2 else None


def evaluate_gates(task_type: str, metrics: dict) -> list:
    """Returns the reasons a candidate fails, empty when it passes every gate."""
    failures = []
    latency = metrics.get("inference_latency_ms")
    if latency is None:
        failures.append("inference latency was not measured")
    elif latency >= MAX_INFERENCE_LATENCY_MS:
        failures.append(f"inference latency {latency} ms is not under {MAX_INFERENCE_LATENCY_MS} ms")
    if not metrics.get("onnx_exported"):
        failures.append(f"no candidate ONNX at {CANDIDATE_ONNX_PATH}")
    if task_type == "CLASSIFICATION":
        fairness = metrics.get("fairness_score")
        if fairness is None:
            failures.append(f"fairness could not be measured on '{FAIRNESS_ATTRIBUTE}' "
                            f"(needs 2+ slices of {MIN_FAIRNESS_SLICE_ROWS}+ validation rows)")
        elif fairness < MIN_FAIRNESS_SCORE:
            failures.append(f"fairness {fairness} on '{FAIRNESS_ATTRIBUTE}' is under {MIN_FAIRNESS_SCORE}")
    return failures


def _ml_os_sandbox(code_string: str, queue: multiprocessing.Queue):
    # 🛡️ BULLETPROOF SAFETY STRIP: Clean it one last time before exec()
    code_string = extract_and_sanitize_code(code_string)

    import warnings
    import math
    warnings.filterwarnings("ignore")
    span_ctx = contextlib.nullcontext()
    if os.getenv("LOGFIRE_TOKEN"):
        try:
            logfire.configure(send_to_logfire="if-token-present")
            span_ctx = logfire.span("sandbox.mle_execution")
        except ImportError:
            pass

    try:
        with span_ctx:
            os.makedirs("models", exist_ok=True)

            if os.path.exists(CANDIDATE_ONNX_PATH):
                os.remove(CANDIDATE_ONNX_PATH)

            unified_namespace = {"__name__": "__main__",
                                 "__builtins__": __builtins__}
            exec(code_string, unified_namespace)

            cand_f1 = unified_namespace.get("candidate_f1")
            cand_rmse = unified_namespace.get("candidate_rmse")
            cand_r2 = unified_namespace.get("candidate_r2")
            cand_silhouette = unified_namespace.get("candidate_silhouette")
            candidate_roc_auc = unified_namespace.get("candidate_roc_auc")
            cand_pr_auc = unified_namespace.get("candidate_pr_auc")
            avg_latency = unified_namespace.get("avg_inference_latency_ms")

            # No fallback to the champion's file: a candidate that didn't export fails its gate.
            onnx_exists = os.path.exists(CANDIDATE_ONNX_PATH)

            def is_valid(val):
                return val is not None and not (isinstance(val, float) and math.isnan(val))

            lat_val = round(float(avg_latency), 4) if is_valid(avg_latency) else None

            if is_valid(cand_silhouette):
                task_type = "CLUSTERING"
                primary_score = round(float(cand_silhouette), 4)
                metrics_payload = {
                    "silhouette_score": primary_score,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists,
                }
            elif is_valid(cand_rmse):
                task_type = "REGRESSION"
                primary_score = round(float(cand_rmse), 4)
                metrics_payload = {
                    "rmse": primary_score,
                    "r2_score": round(float(cand_r2), 4) if is_valid(cand_r2) else None,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists,
                }
            elif is_valid(cand_f1) or is_valid(cand_pr_auc):
                task_type = "CLASSIFICATION"
                primary_score = round(
                    float(cand_pr_auc if is_valid(cand_pr_auc) else cand_f1), 4)
                # Measured here, not taken from the model-written script.
                fairness = compute_fairness(unified_namespace.get("X_val"),
                                            unified_namespace.get("y_val"),
                                            unified_namespace.get("y_pred"))
                metrics_payload = {
                    "pr_auc": primary_score,
                    "f1_score": round(float(cand_f1), 4) if is_valid(cand_f1) else None,
                    "roc_auc": round(float(candidate_roc_auc), 4) if is_valid(candidate_roc_auc) else None,
                    "fairness_score": round(fairness, 4) if fairness is not None else None,
                    "fairness_attribute": FAIRNESS_ATTRIBUTE,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists,
                }
            else:
                queue.put({"status": "MLE_FAILED",
                          "error": "AssertionError: No valid metric."})
                return

            # The registry is not touched here: promotion happens in the auditor, after the audit.
            gate_failures = evaluate_gates(task_type, metrics_payload)
            if gate_failures:
                print(f"[ML Sandbox] 🛑 Candidate fails gates: {'; '.join(gate_failures)}")

            queue.put({
                "status": "ML_TRAINING_SUCCESS",
                "candidate_f1": primary_score if task_type == "CLASSIFICATION" else None,
                "primary_score": primary_score,
                "task_type": task_type,
                "gate_failures": gate_failures,
                "metrics": {"task_type": task_type, **metrics_payload},
            })
    except Exception as e:
        queue.put({"status": "MLE_FAILED",
                  "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        try:
            logfire.force_flush()
        except Exception:
            pass


async def run_ml_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    trace_id = state.get("trace_id", "unknown")
    await publish_stage(trace_id, "ml_executor", "RUNNING")
    result = await _run_ml_executor_node_impl(state)

    # 🛡️ THE FIX: Forcefully increment escalation count right before routing!
    if result.get("status") == "MLE_FAILED":
        retries = state.get("mle_retry_count", 0)
        if retries >= 3:
            result["de_escalation_count"] = state.get("de_escalation_count", 0) + 1

    ok = result.get("status") == "ML_EXECUTION_COMPLETED"
    await publish_stage(
        trace_id, "ml_executor", "SUCCESS" if ok else "FAILED",
        detail=result.get("mle_error_feedback") or "Model trained & exported to ONNX",
    )
    return result


async def _run_ml_executor_node_impl(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    print(
        f"\n{'='*80}\n [NODE 5/5] STARTING ML COMPUTE EXECUTOR\n{'='*80}\n", flush=True)

    code_to_run = state.get("mle_output", "")
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_ml_os_sandbox, args=(code_to_run, queue))
    process.start()
    process.join(timeout=180.0)

    if process.is_alive():
        process.kill()
        process.join()
        return {"status": "MLE_FAILED", "mle_error_feedback": "TimeoutError: Training exceeded 180s.", "mle_previous_code": code_to_run}

    if not queue.empty():
        res = queue.get()
        print(f"[Agent: ML Executor] Sandbox result: {res}", flush=True)

        if res.get("status") == "ML_TRAINING_SUCCESS":
            return {
                "status": "ML_EXECUTION_COMPLETED",
                "mle_error_feedback": None,
                "mle_previous_code": None,
                "candidate_f1": res.get("candidate_f1"),
                "primary_score": res.get("primary_score"),
                "candidate_task_type": res.get("task_type"),
                "gate_failures": res.get("gate_failures"),
                "promotion_status": None,
                "metrics": res.get("metrics"),
                "candidate_metrics": res.get("metrics")
            }
        else:
            return {
                "status": "MLE_FAILED",
                "mle_error_feedback": res.get("error", "Unknown Sandbox Error"),
                "mle_previous_code": code_to_run
            }

    return {"status": "MLE_FAILED", "mle_error_feedback": "No response from sandbox process"}

# ================================================================================
# NODE 6: THE AUDITOR (GOVERNANCE & DLQ ROUTING)
# ================================================================================


async def run_auditor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Central Nervous System: Evaluates trace failures, breaks infinite loops,
    runs AI compliance reviews, and validates sandbox promotions before finalizing.
    """
    trace_id = state.get("trace_id", "unknown")

    # 🌟 Wrap in Logfire span so it renders as a distinct clickable block in observability tools
    with logfire.span("auditor.governance_gate", trace_id=trace_id):
        await publish_stage(trace_id, "auditor_node", "RUNNING", detail="Auditor node active...")

        de_retries = state.get("de_retry_count", 0)
        mle_retries = state.get("mle_retry_count", 0)
        current_status = state.get("status")

        # 🛡️ 1. HYBRID MODEL VALIDATION & GOVERNANCE
        # Checked before the retry limit: a run that trained on its last allowed attempt is
        # audited, not dead-lettered. This is also the only place the registry changes.
        if current_status == "ML_EXECUTION_COMPLETED":
            # 🌟 FIX: Pull correctly from candidate_metrics where the sandbox put it
            ml_metrics = state.get(
                "candidate_metrics") or state.get("metrics") or {}
            psi_score = state.get("metrics", {}).get("psi_score", 0.0)
            task_type = state.get("candidate_task_type") or ml_metrics.get("task_type")
            primary_score = state.get("primary_score", state.get("candidate_f1"))
            gate_failures = state.get("gate_failures")
            if gate_failures is None:
                gate_failures = evaluate_gates(task_type, ml_metrics)
            passed_gates = not gate_failures

            will_promote = _would_promote(task_type, primary_score, passed_gates,
                                          _load_registry(REGISTRY_PATH))

            # --- A. AI Qualitative Review (The New Crew) ---
            # Free-tier budget: only a promotion gets an AI release note. A challenger was
            # already rejected by the deterministic gates, so its note is written from the metrics.
            if will_promote:
                print(
                    "\n[Auditor] 🧠 Waking up AI Compliance Board to review sandbox results...", flush=True)
                from src.agents.crews import get_auditor_crew

                extracted_features = ['account_id',
                                      'transaction_amount', 'category']

                try:
                    auditor_crew = get_auditor_crew(
                        trace_id, psi_score, ml_metrics, extracted_features)
                    release_notes = await auditor_crew.kickoff_async()

                    # Extract the raw markdown text from the Crew output
                    notes_str = release_notes.raw if hasattr(
                        release_notes, 'raw') else str(release_notes)
                except Exception as e:
                    # A provider error (e.g. a spent quota) must not crash the run.
                    notes_str = f"AI release note unavailable ({type(e).__name__}: {str(e)[:200]})."
            else:
                metrics_str = ", ".join(f"{k}={v}" for k, v in ml_metrics.items())
                reason = "; ".join(gate_failures) if gate_failures else "did not beat the champion"
                notes_str = (f"Not promoted ({reason}); no AI review requested. "
                             f"PSI {psi_score}. Metrics: {metrics_str}.")

            print("\n================= 📝 MODEL RELEASE NOTES =================")
            print(notes_str)
            print(
                "==========================================================\n", flush=True)

            # --- B. Deterministic Gate Enforcement ---
            print(
                "[Auditor] 🧱 Recording the candidate against the hard mathematical gates...", flush=True)
            registry = _update_model_registry(
                task_type, ml_metrics, primary_score,
                CANDIDATE_ONNX_PATH, CHAMPION_ONNX_PATH, REGISTRY_PATH, passed_gates)
            promoted = registry["active_champion"] is not None and \
                registry["active_champion"]["version"] == registry["history"][-1]["version"]

            if promoted:
                msg = f"Deterministic Gate Approved: Math checks passed. Promoted to PRODUCTION with score {primary_score}."
                print(f"✅ {msg}", flush=True)

                combined_detail = f"**🤖 AI Compliance Board Report:**\n\n{notes_str}\n\n---\n\n✅ **{msg}**"
                await publish_stage(trace_id, "auditor_node", "SUCCESS", detail=combined_detail)

                # 🌟 FIX: Return metrics along with status so they persist in the final state
                return {
                    "status": "AUDIT_PASSED_PROMOTED",
                    "promotion_status": "PRODUCTION",
                    "primary_score": primary_score,
                    "candidate_metrics": ml_metrics
                }
            else:
                reason = "; ".join(gate_failures) if gate_failures else "did not beat the champion"
                msg = f"Not promoted ({reason}). Logged as CHALLENGER with score {primary_score}."
                print(f"⚠️ {msg}", flush=True)

                combined_detail = f"**🤖 AI Compliance Board Report:**\n\n{notes_str}\n\n---\n\n⚠️ **{msg}**"
                await publish_stage(trace_id, "auditor_node", "SUCCESS", detail=combined_detail)

                # 🌟 FIX: Return metrics along with status so they persist in the final state
                return {
                    "status": "AUDIT_PASSED_CHALLENGER",
                    "promotion_status": "CHALLENGER",
                    "primary_score": primary_score,
                    "candidate_metrics": ml_metrics
                }

        # 🛡️ 2. INFINITE LOOP PROTECTION (DEAD-LETTER QUEUE)
        if de_retries >= 3 or mle_retries >= 3:
            dlq_reason = f"Max retries exceeded (DE: {de_retries}, MLE: {mle_retries})."
            print(
                f"\n[Auditor] 🚨 {dlq_reason} Routing trace {trace_id} to NATS Dead-Letter Queue (DLQ).", flush=True)

            await publish_stage(trace_id, "auditor_node", "FAILED", detail="Routed to DLQ for human review.")
            return {"status": "DLQ_ROUTED", "auditor_feedback": dlq_reason}

        # 🛡️ 3. DATA ENGINEERING VALIDATION
        if current_status == "SUCCESS":  # Meaning DE sandbox passed
            print(
                f"\n[Auditor] 🧹 Data patch validated. Approving route to MLE Profiler.", flush=True)
            await publish_stage(trace_id, "auditor_node", "SUCCESS", detail="Data patch validated.")
            return {"status": "AUDIT_PASSED_DATA"}

        await publish_stage(trace_id, "auditor_node", "SUCCESS", detail="Auditor bypassed.")
        return {"status": "AUDITOR_BYPASSED"}
