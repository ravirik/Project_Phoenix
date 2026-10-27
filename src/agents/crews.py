import os
import logfire
from crewai import Agent, Crew, Process, Task
from src.config.llm import get_crewai_llm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def agent_step_callback(agent_completion):
    agent_name = getattr(agent_completion, "agent", "Specialized Agent")
    raw_output = str(getattr(agent_completion, "output", agent_completion))
    logfire.info("🤖 {agent_name} Active", agent_name=str(
        agent_name), step_output=raw_output[:300])


def create_de_crew(de_error_feedback: str = None, previous_code: str = None) -> Crew:
    llm = get_crewai_llm()

    # Free-tier budget: one agent with no tools makes exactly one model request.
    # The playbook search runs before the crew (see _run_de_crew_node_impl), so the
    # agent doesn't spend requests on a tool-calling loop, and the separate SRE
    # diagnosis step is gone because nothing downstream read its report.
    de_agent = Agent(
        role="Data Engineer",
        goal="Formulate an executable Python Pandas patch to fix data anomalies in-place.",
        backstory="You are a principal Data Engineer. Output ONLY valid, executable Pandas code enclosed inside triple backticks.",
        llm=llm, max_iter=1, verbose=True
    )

    patch_description = (
        "Trace ID '{trace_id}': telemetry alerts indicate schema drift and data quality degradation.\n"
        "Formulate an in-place Python Pandas patch to clean the DataFrame 'df'.\n"
        "ACTUAL DATAFRAME COLUMNS IN DATASET: {schema_columns}\n"
        "COLUMN DTYPES AND NULL COUNTS: {schema_profile}\n"
        "CLOSEST HISTORICAL PLAYBOOK (reuse only if it fits these columns):\n{playbook_context}\n"
        "Requirements:\n"
        "1. Use ONLY the columns listed above in your pandas operations to avoid KeyErrors.\n"
        "2. Output ONLY the Python patch strictly inside ```python ... ``` triple backticks.\n"
        "3. Clean 'df' directly: convert string columns to numeric using pd.to_numeric(..., errors='coerce'), "
        "and use .fillna(0) instead of dropping rows. NEVER drop all rows or output an empty DataFrame.\n"
        "4. Do NOT include file read/write operations (no read_parquet/to_parquet)."
    )

    if de_error_feedback and previous_code:
        # 🛡️ SMART TRUNCATION: Only send the tail end of the code
        truncated_code = previous_code[-1000:] if len(previous_code) > 1000 else previous_code
        patch_description += (
            f"\n\n🚨 CRITICAL FAILURE IN PREVIOUS ATTEMPT 🚨\n"
            f"Your previous code ended with:\n```python\n...\n{truncated_code}\n```\n"
            f"Failed in the sandbox with error:\n{de_error_feedback}\n"
            f"DO NOT REPEAT THE SAME MISTAKE. Analyze the error and provide the FULL, corrected Python code."
        )

    task_patch = Task(
        description=patch_description,
        expected_output="Executable Python Pandas snippet inside triple backticks ```python ... ```.",
        agent=de_agent
    )

    return Crew(
        agents=[de_agent],
        tasks=[task_patch],
        process=Process.sequential,
        step_callback=agent_step_callback,
        verbose=True
    )


def create_mle_crew(mle_error_feedback: str = None, previous_code: str = None) -> Crew:
    llm = get_crewai_llm()

    mle_agent = Agent(
        role="MLOps Release Engineer",
        goal="Train, evaluate, and export machine learning models dynamically across Classification, Regression, and Clustering tasks.",
        backstory=(
            "You are a principal MLOps Release Engineer. You write self-contained Python scripts "
            "to fit Scikit-Learn models tailored to the auto-detected task type, evaluate performance, "
            "export candidate models to ONNX format, and output metric variables into top-level scope."
        ),
        llm=llm, max_iter=1, verbose=True
    )

    serialize_description = (
        "Write a standalone Python script to execute end-to-end Model Training, Evaluation, and ONNX Export.\n\n"
        "CONFIG:\n"
        "- Dataset Path: '{data_path}'\n"
        "- Task Type: '{task_type}'\n"
        "- Target Column: '{target_col}'\n"
        "- Column dtypes: {schema_profile}\n"
        "- Text columns load as pandas StringDtype, which NumPy can't interpret. Detect them with "
        "pd.api.types.is_numeric_dtype, never np.issubdtype or dtype == object.\n\n"
        "REQUIREMENTS:\n"
        "1. DATA LOADING & PREPROCESSING:\n"
        "   - Load Parquet dataset strictly from '{data_path}'.\n"
        "   - IF task_type is 'CLUSTERING': Use all numeric/encoded features as feature matrix X (no target y).\n"
        "   - ELSE: Separate feature matrix X and target y = df['{target_col}']. Perform an 80/20 Train/Val split (test_size=0.2, random_state=42).\n"
        "   - Use ColumnTransformer with StandardScaler for numerical features and OneHotEncoder(handle_unknown='ignore') for string categoricals.\n\n"
        "2. MODEL TRAINING & METRICS BY TASK TYPE:\n"
        "   - IF task_type == 'REGRESSION':\n"
        "     Fit sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42).\n"
        "     Compute candidate_rmse = sqrt(mean_squared_error(y_val, y_pred)) and candidate_r2 = r2_score(y_val, y_pred).\n\n"
        "   - IF task_type == 'CLUSTERING':\n"
        "     Fit sklearn.cluster.KMeans(n_clusters=3, random_state=42, n_init=10) on X.\n"
        "     Compute candidate_silhouette = silhouette_score(preprocessor.fit_transform(X), model.labels_).\n\n"
        "   - IF task_type in ['BINARY_CLASSIFICATION', 'MULTICLASS_CLASSIFICATION']:\n"
        "     Fit sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42).\n"
        "     Compute candidate_f1 = f1_score(y_val, y_pred, average='weighted').\n"
        "     Calculate y_prob = pipeline.predict_proba(X_val).\n"
        "     Compute candidate_roc_auc = roc_auc_score(y_val, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, multi_class='ovr').\n"
        "     from sklearn.metrics import average_precision_score\n"
        "     Compute candidate_pr_auc = average_precision_score(y_val, y_prob[:, 1]) if y_prob.shape[1] == 2 else candidate_f1.\n"
        "     Do not compute fairness: the sandbox measures it. Keep X_val (the raw validation DataFrame, before encoding), y_val and y_pred (predictions on X_val) as top-level variables.\n\n"
        "3. LATENCY PROFILING (ALL TASKS):\n"
        "   - Import the 'time' module.\n"
        "   - Calculate avg_inference_latency_ms by calling time.time() before and after pipeline.predict(X_val), computing the difference, dividing by len(X_val), and multiplying by 1000 to get milliseconds per sample.\n\n"
        "4. ONNX EXPORT ONLY:\n"
        "   - Save the fitted pipeline/model to ONNX format at 'models/candidate_model.onnx' using skl2onnx.\n"
        "   - CRITICAL: If exporting a classifier, you MUST pass options={id(model): {'zipmap': False}} to convert_sklearn to prevent dictionary serialization errors.\n"
        "   - Explicitly assign metric variables (candidate_f1, candidate_roc_auc, candidate_pr_auc, avg_inference_latency_ms, candidate_rmse, candidate_r2, candidate_silhouette) in top-level script scope.\n\n"
        "5. OUTPUT FORMAT (CRITICAL):\n"
        "   - Output strictly inside a single ```python ... ``` triple backtick block."
    )

    if mle_error_feedback and previous_code:
        # 🛡️ SMART TRUNCATION: Only send the tail end of the code
        truncated_code = previous_code[-1000:] if len(previous_code) > 1000 else previous_code
        serialize_description += (
            f"\n\n🚨 CRITICAL FAILURE IN PREVIOUS ATTEMPT 🚨\n"
            f"Your previous code ended with:\n```python\n...\n{truncated_code}\n```\n"
            f"Failed in the sandbox with error:\n{mle_error_feedback}\n"
            f"DO NOT REPEAT THE SAME MISTAKE. Analyze the error and provide the FULL, corrected Python code."
        )

    task_serialize = Task(
        description=serialize_description,
        expected_output="A standalone Python script enclosed strictly inside triple backticks ```python ... ```.",
        agent=mle_agent
    )

    return Crew(agents=[mle_agent], tasks=[task_serialize], process=Process.sequential, step_callback=agent_step_callback, verbose=True)


def is_rate_limit(exception):
    return "429" in str(exception) or "Resource exhausted" in str(exception)


def is_daily_quota_exhausted(exception) -> bool:
    """A per-day quota (the free tier's) won't reset for hours, so retrying only burns time."""
    message = str(exception)
    return "RESOURCE_EXHAUSTED" in message and "PerDay" in message


def _worth_retrying(exception) -> bool:
    return not is_daily_quota_exhausted(exception)


@retry(
    wait=wait_exponential(multiplier=15, min=30, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_worth_retrying),
    reraise=True
)
async def run_de_crew_instrumented(inputs: dict, err_fb: str = None, prev_code: str = None) -> str:
    crew = create_de_crew(err_fb, prev_code)
    with logfire.span("crewai.de_crew", trace_id=inputs.get("trace_id")):
        res = await crew.kickoff_async(inputs=inputs)
        return str(res)


@retry(
    wait=wait_exponential(multiplier=15, min=30, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_worth_retrying),
    reraise=True
)
async def run_mle_crew_instrumented(inputs: dict, err_fb: str = None, prev_code: str = None) -> str:
    crew = create_mle_crew(err_fb, prev_code)
    with logfire.span("crewai.mle_crew", data_path=inputs.get("data_path")):
        res = await crew.kickoff_async(inputs=inputs)
        return getattr(res, "raw", str(res))


def get_auditor_crew(trace_id: str, psi_score: float, metrics: dict, features: list):
    """
    Hybrid Governance: The AI reads the sandbox output and writes a compliance report 
    BEFORE the deterministic Python gate makes the final push to production.
    """

    # Free-tier budget: one agent, one task, one model request. The compliance verdict
    # and the release note used to be two agents (four requests including hand-offs).
    # llm= is set explicitly; without it CrewAI falls back to OpenAI.
    release_manager = Agent(
        role="Lead MLOps Release Manager and Compliance Reviewer",
        goal="Check the candidate's features for bias proxies, then write the release note.",
        backstory=(
            "You review ML releases for a major financial institution. You flag features that act as "
            "protected demographics or proxies for bias, and you translate drift (PSI) and model metrics "
            "(F1, PR-AUC) into clear, executive-friendly release notes. You do not write code."
        ),
        llm=get_crewai_llm(),
        max_iter=1,
        verbose=True,
        allow_delegation=False
    )

    # 🛠️ THE FIX: Format the dictionary cleanly into a readable Markdown bulleted list
    metrics_str = "\n".join([f"- **{k.upper()}**: {v}" for k, v in metrics.items()]
                            ) if metrics else "No sandbox metrics recorded."

    release_notes_task = Task(
        description=(
            f"Write a formal 'Automated Model Release Note' for trace {trace_id}.\n"
            f"Context: The system detected Data Drift with a PSI score of {psi_score}.\n"
            f"Candidate model features: {features}.\n"
            f"Sandbox Evaluation Metrics:\n{metrics_str}\n\n"
            "Include the following sections:\n"
            "1. 🚨 Incident Trigger (Why did we retrain? Mention the PSI score).\n"
            "2. 🛡️ Compliance Verdict (Flag any feature that could proxy for demographic bias, e.g. zip codes "
            "or names. If none, state 'COMPLIANCE PASSED: No high-risk features detected.' and justify in 2 sentences).\n"
            "3. 📊 Sandbox Performance (List the PR-AUC, Latency, and Fairness scores).\n"
            "4. 🤖 AI Recommendation (State whether you recommend deploying this model based on the metrics)."
        ),
        expected_output="A markdown-formatted Release Note ready for a Slack/Teams channel.",
        agent=release_manager
    )

    return Crew(
        agents=[release_manager],
        tasks=[release_notes_task],
        process=Process.sequential
    )
