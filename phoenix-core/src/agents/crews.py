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
        goal="Evaluate candidate model metrics, benchmark against production Champion, and export ONNX.",
        backstory="You are a principal MLOps Release Engineer. You write standalone Python scripts to evaluate candidate models and export ONNX graphs.",
        llm=llm, max_iter=3, verbose=True
    )

    serialize_description = (
        "Write a standalone Python script to execute end-to-end Feature Engineering, Model Training, Validation, and ONNX Export.\n"
        "Requirements:\n"
        "1. FEATURE ENGINEERING & SPLIT:\n"
        "   - Load parquet dataset from '{data_path}'.\n"
        "   - Separate features (X) and target label 'target' (y).\n"
        "   - Apply StandardScaler/MinMaxScaler or handle categorical encodings on X.\n"
        "   - Perform an 80/20 Train/Validation split using train_test_split(test_size=0.2, random_state=42).\n\n"
        "2. MODEL TRAINING (EPOCHS / FITTING):\n"
        "   - Train a Scikit-Learn (e.g., RandomForest/LogisticRegression) or PyTorch Neural Network.\n"
        "   - If PyTorch: Train for 10 epochs using Adam optimizer and BCE/CrossEntropy loss.\n\n"
        "3. MODEL VALIDATION & BENCHMARKING:\n"
        "   - Evaluate the candidate model on the 20% Validation Set.\n"
        "   - Compute Validation F1-Score, ROC-AUC, and average inference latency (ms over 100 iterations).\n"
        "   - Compare Validation F1 against Champion F1 in 'models/model_registry.json' (default Champ F1 = 0.80).\n"
        "   - Set promotion_status to 'PRODUCTION' if candidate F1 > Champion F1, else 'CHALLENGER'.\n\n"
        "4. EXPORT & REGISTRY:\n"
        "   - Export the trained pipeline/model to ONNX format at 'models/pipeline_model.onnx'.\n"
        "   - Update 'models/model_registry.json' with candidate metrics and status.\n"
        "5. Output ONLY the Python script strictly enclosed inside ```python ... ``` triple backticks."
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
    # Waits 30s, then 60s, then 120s
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
