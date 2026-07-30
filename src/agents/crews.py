import os
import logfire
from crewai import Agent, Crew, Process, Task
from src.config.llm import get_crewai_llm
from src.agents.tools import search_historical_playbooks
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def agent_step_callback(agent_completion):
    agent_name = getattr(agent_completion, "agent", "Specialized Agent")
    raw_output = str(getattr(agent_completion, "output", agent_completion))
    logfire.info("🤖 {agent_name} Active", agent_name=str(
        agent_name), step_output=raw_output[:300])


def create_de_crew(de_error_feedback: str = None, previous_code: str = None) -> Crew:
    llm = get_crewai_llm()

    sre_agent = Agent(
        role="Data Quality SRE",
        goal="Diagnose pipeline telemetry anomalies and schema drift.",
        backstory="You are a senior Data SRE. Provide concise diagnostic reports based on telemetry logs.",
        llm=llm, max_iter=3, verbose=True
    )

    de_agent = Agent(
        role="Data Engineer",
        goal="Formulate an executable Python Pandas patch to fix data anomalies in-place.",
        backstory="You are a principal Data Engineer. Output ONLY valid, executable Pandas code enclosed inside triple backticks.",
        tools=[search_historical_playbooks],
        llm=llm, max_iter=3, verbose=True
    )

    task_diagnose = Task(
        description="Analyze trace ID '{trace_id}'. Telemetry alerts indicate schema drift and data quality degradation. Summarize the defect.",
        expected_output="An incident diagnostic report.",
        agent=sre_agent
    )

    patch_description = (
        "Formulate an in-place Python Pandas patch to clean the DataFrame 'df'.\n"
        "ACTUAL DATAFRAME COLUMNS IN DATASET: {schema_columns}\n"
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
        agents=[sre_agent, de_agent],
        tasks=[task_diagnose, task_patch],
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
        llm=llm, max_iter=3, verbose=True
    )

    serialize_description = (
        "Write a standalone Python script to execute end-to-end Model Training, Evaluation, and ONNX Export.\n\n"
        "CONFIG:\n"
        "- Dataset Path: '{data_path}'\n"
        "- Task Type: '{task_type}'\n"
        "- Target Column: '{target_col}'\n\n"
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
        "     Calculate candidate_fairness: Check if 'categorical_features' list is not empty. If true, pick the first categorical feature, iterate through its unique values in X_val, compute the f1_score for each slice, and set candidate_fairness to the minimum score. If empty, set candidate_fairness = 1.0.\n\n"
        "3. LATENCY PROFILING (ALL TASKS):\n"
        "   - Import the 'time' module.\n"
        "   - Calculate avg_inference_latency_ms by calling time.time() before and after pipeline.predict(X_val), computing the difference, dividing by len(X_val), and multiplying by 1000 to get milliseconds per sample.\n\n"
        "4. ONNX EXPORT ONLY:\n"
        "   - Save the fitted pipeline/model to ONNX format at 'models/candidate_model.onnx' using skl2onnx.\n"
        "   - CRITICAL: If exporting a classifier, you MUST pass options={id(model): {'zipmap': False}} to convert_sklearn to prevent dictionary serialization errors.\n"
        "   - Explicitly assign metric variables (candidate_f1, candidate_roc_auc, candidate_pr_auc, candidate_fairness, avg_inference_latency_ms, candidate_rmse, candidate_r2, candidate_silhouette) in top-level script scope.\n\n"
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


@retry(
    wait=wait_exponential(multiplier=15, min=30, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
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
    retry=retry_if_exception_type(Exception),
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

    # --- AGENTS ---
    compliance_officer = Agent(
        role="AI Ethics & Compliance Officer",
        goal="Ensure the newly trained candidate model does not rely on biased features and complies with enterprise fairness standards.",
        backstory=(
            "You are a strict compliance officer for a major financial institution. "
            "Your job is to review the features used by ML models to ensure they don't use "
            "protected demographics or proxies for bias. You do not write code; you write audits."
        ),
        verbose=True,
        allow_delegation=False
    )

    mlops_lead = Agent(
        role="Lead MLOps Release Manager",
        goal="Synthesize drift metrics, compliance reports, and model performance into a final Release Note.",
        backstory=(
            "You are the Lead MLOps Engineer bridging the gap between machines and business stakeholders. "
            "You translate complex statistical drift (PSI, KL-Divergence) and raw model metrics (F1, PR-AUC) "
            "into clear, executive-friendly release notes."
        ),
        verbose=True,
        allow_delegation=False
    )

    # 🛠️ THE FIX: Format the dictionary cleanly into a readable Markdown bulleted list
    metrics_str = "\n".join([f"- **{k.upper()}**: {v}" for k, v in metrics.items()]
                            ) if metrics else "No sandbox metrics recorded."

    # --- TASKS ---
    compliance_task = Task(
        description=(
            f"Review the following feature list used by the new candidate model: {features}.\n"
            "1. Flag any features that could be considered proxies for demographic bias (e.g., zip codes, names).\n"
            "2. If no risky features are found, explicitly state 'COMPLIANCE PASSED: No high-risk features detected.'\n"
            "3. Provide a brief 2-sentence justification for your decision."
        ),
        expected_output="A short compliance verdict passing or failing the feature list.",
        agent=compliance_officer
    )

    release_notes_task = Task(
        description=(
            f"Write a formal 'Automated Model Release Note' for trace {trace_id}.\n"
            f"Context: The system detected Data Drift with a PSI score of {psi_score}.\n"
            # 👈 Dynamic injected metrics!
            f"Sandbox Evaluation Metrics:\n{metrics_str}\n\n"
            "Include the following sections:\n"
            "1. 🚨 Incident Trigger (Why did we retrain? Mention the PSI score).\n"
            "2. 🛡️ Compliance Verdict (Read the output from the Compliance Officer).\n"
            "3. 📊 Sandbox Performance (List the PR-AUC, Latency, and Fairness scores).\n"
            "4. 🤖 AI Recommendation (State whether you recommend deploying this model based on the metrics)."
        ),
        expected_output="A markdown-formatted Release Note ready for a Slack/Teams channel.",
        agent=mlops_lead
    )

    # --- CREW ---
    return Crew(
        agents=[compliance_officer, mlops_lead],
        tasks=[compliance_task, release_notes_task],
        process=Process.sequential
    )