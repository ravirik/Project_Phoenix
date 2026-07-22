import os
import logfire
from crewai import Agent, Crew, Process, Task
from src.config.llm import get_crewai_llm
from src.agents.tools import search_historical_playbooks
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def agent_step_callback(agent_completion):
    """Real-time observability into active agent thoughts and tool actions."""
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
        "3. Clean 'df' directly: replace empty strings with NaN, convert types, and drop null rows.\n"
        "4. Do NOT include file read/write operations (no read_parquet/to_parquet)."
    )

    # 🛡️ DYNAMIC REVALIDATION GUARDRAIL
    if de_error_feedback and previous_code:
        patch_description += (
            f"\n\n🚨 CRITICAL FAILURE IN PREVIOUS ATTEMPT 🚨\n"
            f"Your previous code:\n```python\n{previous_code}\n```\n"
            f"Failed in the sandbox with error:\n{de_error_feedback}\n"
            f"DO NOT REPEAT THE SAME MISTAKE. Analyze the error and provide the corrected Python code."
        )

    task_patch = Task(
        description=patch_description,
        expected_output="Executable Python Pandas snippet inside triple backticks ```python ... ```.",
        agent=de_agent
    )

    return Crew(agents=[sre_agent, de_agent], tasks=[task_diagnose, task_patch], process=Process.sequential, step_callback=agent_step_callback, verbose=True)


def create_mle_crew(mle_error_feedback: str = None, previous_code: str = None) -> Crew:
    llm = get_crewai_llm()

    mle_agent = Agent(
        role="MLOps Release Engineer",
        goal="Train, evaluate, and export machine learning models dynamically across Classification, Regression, and Clustering tasks.",
        backstory=(
            "You are a principal MLOps Release Engineer. You write self-contained Python scripts "
            "to fit Scikit-Learn models tailored to the auto-detected task type, evaluate performance, "
            "export models to ONNX format, and update local model registries."
        ),
        llm=llm, max_iter=3, verbose=True
    )

    serialize_description = (
        "Write a standalone Python script to execute end-to-end Model Training, Evaluation, ONNX Export, and Registry Tracking.\n\n"
        "CONFIG:\n"
        "- Dataset Path: '{data_path}'\n"
        "- Task Type: '{task_type}'\n"
        "- Target Column: '{target_col}'\n\n"
        "REQUIREMENTS:\n"
        "1. DATA LOADING & PREPROCESSING:\n"
        "   - Load Parquet dataset from '{data_path}'.\n"
        "   - IF task_type is 'CLUSTERING': Use all numeric/encoded features as feature matrix X (no target y).\n"
        "   - ELSE: Separate feature matrix X and target y = df['{target_col}']. Perform an 80/20 Train/Val split (test_size=0.2, random_state=42).\n"
        "   - Use ColumnTransformer with StandardScaler for numerical features and OneHotEncoder(handle_unknown='ignore') for string categoricals.\n\n"
        "2. MODEL TRAINING & METRICS BY TASK TYPE:\n"
        "   - IF task_type == 'REGRESSION':\n"
        "     Fit sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42).\n"
        "     Compute candidate_rmse = root_mean_squared_error(y_val, y_pred) and candidate_r2 = r2_score(y_val, y_pred).\n"
        "     Set promotion_status = 'PRODUCTION' if candidate_rmse < 10.0 else 'CHALLENGER'.\n\n"
        "   - IF task_type == 'CLUSTERING':\n"
        "     Fit sklearn.cluster.KMeans(n_clusters=3, random_state=42) on X.\n"
        "     Compute candidate_silhouette = silhouette_score(X, model.labels_).\n"
        "     Set promotion_status = 'PRODUCTION' if candidate_silhouette > 0.35 else 'CHALLENGER'.\n\n"
        "   - IF task_type in ['BINARY_CLASSIFICATION', 'MULTICLASS_CLASSIFICATION']:\n"
        "     Fit sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42).\n"
        "     Compute candidate_f1 = f1_score(y_val, y_pred, average='weighted').\n"
        "     Set promotion_status = 'PRODUCTION' if candidate_f1 > 0.80 else 'CHALLENGER'.\n\n"
        "3. EXPORT & REGISTRY:\n"
        "   - Save the fitted pipeline/model to ONNX format at 'models/pipeline_model.onnx' using skl2onnx.\n"
        "   - Update 'models/model_registry.json' with metrics and promotion_status.\n"
        "   - Explicitly assign metric variables (candidate_f1, candidate_rmse, candidate_silhouette, promotion_status) in top-level script scope.\n\n"
        "4. OUTPUT FORMAT:\n"
        "   - Output ONLY the Python script strictly enclosed inside ```python ... ``` triple backticks."
    )

    # 🛡️ DYNAMIC REVALIDATION GUARDRAIL
    if mle_error_feedback and previous_code:
        serialize_description += (
            f"\n\n🚨 CRITICAL FAILURE IN PREVIOUS ATTEMPT 🚨\n"
            f"Your previous code:\n```python\n{previous_code}\n```\n"
            f"Failed in the sandbox with error:\n{mle_error_feedback}\n"
            f"DO NOT REPEAT THE SAME MISTAKE. Analyze the error, fix the bug, and provide the corrected Python code."
        )

    task_serialize = Task(
        description=serialize_description,
        expected_output="A standalone Python script enclosed strictly inside triple backticks ```python ... ```.",
        agent=mle_agent
    )

    return Crew(agents=[mle_agent], tasks=[task_serialize], process=Process.sequential, step_callback=agent_step_callback, verbose=True)


# Define a custom retry condition for Rate Limits
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
