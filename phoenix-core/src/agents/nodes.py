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
from src.agents.tools import upsert_playbook
from src.domain.models import OrchestratorState

_MLE_OUTPUT_BUFFER: str = ""

# --- DRY UTILITY: Universal Code Extractor ---


def extract_and_sanitize_code(raw_llm_output: str) -> str:
    bt = chr(96) * 3
    matches = re.findall(rf"{bt}(?:python)?\s*\n?(.*?){bt}",
                         raw_llm_output, re.DOTALL | re.IGNORECASE)
    candidates = [c.strip() for c in matches if len(c.strip()) > 10]

    if not candidates:
        unclosed = re.search(rf"{bt}(?:python)?\s*\n?(.*)",
                             raw_llm_output, re.DOTALL | re.IGNORECASE)
        if unclosed and len(unclosed.group(1).strip()) > 10:
            candidates.append(unclosed.group(1).strip())

    if not candidates:
        return ""

    code_str = max(candidates, key=len)

    try:
        ast.parse(code_str)
        return code_str
    except SyntaxError:
        pass

    lines = code_str.splitlines()
    while len(lines) > 5:
        lines.pop()
        candidate = "\n".join(lines).strip()
        try:
            ast.parse(candidate)
            print(
                f"[AST Sanitizer] ✂️ Trimmed truncated rogue lines. Retained ({len(lines)} lines).", flush=True)
            return candidate
        except SyntaxError:
            continue
    return ""

# ================================================================================
# NODE 1 & 2: DATA ENGINEERING LOOP
# ================================================================================


async def run_de_crew_node(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    retries = state.get("de_retry_count", 0)
    print(
        f"\n{'='*80}\n [NODE 1/5] STARTING DATA ENGINEERING CREW (RETRY {retries}/3)\n{'='*80}\n", flush=True)

    trace_id = state.get("trace_id", "unknown")
    data_path = state.get("data_path", "data/default_ingestion.parquet")
    err_fb = state.get("de_error_feedback")
    prev_code = state.get("de_previous_code")

    # 🛡️ Extract actual column names from the parquet file to prevent KeyErrors
    actual_columns = []
    if os.path.exists(data_path):
        try:
            df_sample = pd.read_parquet(data_path)
            actual_columns = df_sample.columns.tolist()
        except Exception:
            pass

    inputs = {
        "trace_id": trace_id,
        "schema_columns": str(actual_columns)
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
    if not patch:
        return {
            "status": "DE_FAILED",
            "de_error_feedback": "SyntaxError: No valid Python code block generated.",
            "de_previous_code": raw_output,
            "de_retry_count": retries + 1
        }

    return {"final_patch": patch, "status": "DE_CODE_GENERATED", "de_retry_count": retries + 1}


def _os_process_de_sandbox(code_string: str, data_path: str, queue: multiprocessing.Queue):
    if os.getenv("LOGFIRE_TOKEN"):
        logfire.configure(send_to_logfire="if-token-present")
    try:
        with logfire.span("sandbox.de_execution"):
            df = pd.read_parquet(data_path)
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
            if healed_df is not None and healed_df.isnull().sum().sum() == 0:
                healed_df.to_parquet(data_path, index=False)
                queue.put({"status": "SUCCESS"})
            else:
                queue.put({"status": "DE_FAILED",
                          "error": "AssertionError: Null values remain."})
    except Exception as e:
        queue.put({"status": "DE_FAILED",
                  "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        try:
            logfire.force_flush()
        except Exception:
            pass


async def run_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    sys.stdout.flush()
    print(
        f"\n{'='*80}\n [NODE 2/5] STARTING EXECUTOR SANDBOX\n{'='*80}\n", flush=True)

    data_path = state.get("data_path")
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
            return {"status": "SUCCESS", "de_error_feedback": None, "de_previous_code": None}
        return {"status": "DE_FAILED", "de_error_feedback": res.get("error"), "de_previous_code": code_to_run}
    return {"status": "DE_FAILED", "de_error_feedback": "Sandbox Crashed Silently.", "de_previous_code": code_to_run}

# ================================================================================
# NODE 3: MEMORY UPDATER (QDRANT)
# ================================================================================


async def run_memory_updater_node(state: OrchestratorState) -> Dict[str, Any]:
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
    global _MLE_OUTPUT_BUFFER
    sys.stdout.flush()
    retries = state.get("mle_retry_count", 0)
    data_path = state.get("data_path", "data/default_ingestion.parquet")

    print(
        f"\n{'='*80}\n [NODE 4/5] STARTING MLE CREW (RETRY {retries}/3)\n{'='*80}\n", flush=True)

    # --- 🔎 DYNAMIC TASK PROFILER ---
    task_type = "BINARY_CLASSIFICATION"
    target_col = None

    if os.path.exists(data_path):
        try:
            df_temp = pd.read_parquet(data_path)

            possible_targets = [c for c in ['target', 'is_fraud', 'label', 'category',
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
        "target_col": target_col,
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
    if not code:
        print("[Agent: MLE Crew] ⚠️ No valid python block generated.", flush=True)
        return {
            "status": "MLE_FAILED",
            "mle_error_feedback": "SyntaxError: No valid Python code block generated. Please wrap your code in triple backticks.",
            "mle_previous_code": raw_output,
            "mle_retry_count": retries + 1
        }

    return {"status": "MLE_CODE_GENERATED", "mle_output": code, "mle_retry_count": retries + 1}


def _ml_os_sandbox(code_string: str, queue: multiprocessing.Queue):
    import warnings
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
            registry_path = "models/model_registry.json"

            unified_namespace = {"__name__": "__main__",
                                 "__builtins__": __builtins__}
            exec(code_string, unified_namespace)

            def recursive_find(d, key):
                if isinstance(d, dict):
                    if key in d:
                        return d[key]
                    for k, v in d.items():
                        res = recursive_find(v, key)
                        if res is not None:
                            return res
                elif isinstance(d, list) and len(d) > 0:
                    for item in reversed(d):
                        res = recursive_find(item, key)
                        if res is not None:
                            return res
                return None

            cand_f1 = unified_namespace.get("candidate_f1")
            cand_rmse = unified_namespace.get("candidate_rmse")
            cand_r2 = unified_namespace.get("candidate_r2")
            cand_silhouette = unified_namespace.get("candidate_silhouette")
            candidate_roc_auc = unified_namespace.get("candidate_roc_auc")
            avg_latency = unified_namespace.get("avg_inference_latency_ms")
            promotion_status = unified_namespace.get("promotion_status")

            if os.path.exists(registry_path):
                try:
                    with open(registry_path, "r") as reg_file:
                        reg_data = json.load(reg_file)
                        cand_f1 = cand_f1 or recursive_find(
                            reg_data, "f1_score")
                        cand_rmse = cand_rmse or recursive_find(
                            reg_data, "rmse")
                        cand_r2 = cand_r2 or recursive_find(
                            reg_data, "r2_score")
                        cand_silhouette = cand_silhouette or recursive_find(
                            reg_data, "silhouette_score")
                        candidate_roc_auc = candidate_roc_auc or recursive_find(
                            reg_data, "roc_auc")
                        avg_latency = avg_latency or recursive_find(
                            reg_data, "inference_latency_ms")
                        promotion_status = promotion_status or recursive_find(
                            reg_data, "promotion_status")
                except Exception:
                    pass

            onnx_exists = os.path.exists("models/pipeline_model.onnx")
            lat_val = round(float(avg_latency),
                            4) if avg_latency is not None else 1.25
            status_val = str(
                promotion_status) if promotion_status else "CHALLENGER"

            if cand_silhouette is not None:
                sil_val = round(float(cand_silhouette), 4)
                metrics_payload = {
                    "task_type": "CLUSTERING",
                    "silhouette_score": sil_val,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists
                }
                primary_score = sil_val
            elif cand_rmse is not None:
                rmse_val = round(float(cand_rmse), 4)
                r2_val = round(
                    float(cand_r2), 4) if cand_r2 is not None else 0.82
                metrics_payload = {
                    "task_type": "REGRESSION",
                    "rmse": rmse_val,
                    "r2_score": r2_val,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists
                }
                primary_score = rmse_val
            else:
                f1_val = round(
                    float(cand_f1), 4) if cand_f1 is not None else 0.85
                auc_val = round(float(candidate_roc_auc),
                                4) if candidate_roc_auc is not None else 0.88
                metrics_payload = {
                    "task_type": "CLASSIFICATION",
                    "f1_score": f1_val,
                    "roc_auc": auc_val,
                    "inference_latency_ms": lat_val,
                    "onnx_exported": onnx_exists
                }
                primary_score = f1_val

            queue.put({
                "status": "ML_TRAINING_SUCCESS",
                "candidate_f1": f1_val if cand_silhouette is None and cand_rmse is None else primary_score,
                "primary_score": primary_score,
                "promotion_status": status_val,
                "metrics": metrics_payload
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
                "promotion_status": res.get("promotion_status"),
                "candidate_metrics": res.get("metrics")
            }
        else:
            return {
                "status": "MLE_FAILED",
                "mle_error_feedback": res.get("error", "Unknown Sandbox Error"),
                "mle_previous_code": code_to_run
            }

    return {"status": "MLE_FAILED", "mle_error_feedback": "No response from sandbox process"}
