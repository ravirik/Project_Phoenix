import os
import logfire
from crewai import Agent, Crew, Process, Task
from src.config.llm import get_crewai_llm
from src.agents.tools import search_historical_playbooks


def create_de_crew() -> Crew:
    llm = get_crewai_llm()

    sre_agent = Agent(
        role="Data Quality SRE",
        goal="Diagnose pipeline telemetry anomalies and schema drift.",
        backstory="You are a senior Data SRE. You provide concise diagnostic reports based on pipeline telemetry logs.",
        llm=llm,
        max_iter=5,
        max_execution_time=60,
        verbose=True
    )

    de_agent = Agent(
        role="Data Engineer",
        goal="Formulate an executable Python Pandas patch to fix data anomalies in-place.",
        backstory="You are a principal Data Engineer. You ALWAYS write minimal, executable Pandas code enclosed inside triple backticks ```python ... ``` to sanitize the DataFrame 'df'.",
        tools=[search_historical_playbooks],
        llm=llm,
        max_iter=5,
        max_execution_time=60,
        verbose=True
    )

    task_diagnose = Task(
        description=(
            "Analyze trace ID '{trace_id}'. Telemetry alerts indicate schema drift and data quality degradation: "
            "empty string representations, dirty numeric formatting (e.g., '$120.50'), and null records present in ingestion tables. "
            "Summarize the defect."
        ),
        expected_output="An incident diagnostic report detailing the schema drift and data quality failure.",
        agent=sre_agent
    )

    task_patch = Task(
        description=(
            "Formulate an in-place Python Pandas patch to clean the DataFrame 'df'.\n"
            "Requirements:\n"
            "1. Search historical playbooks using search_historical_playbooks tool with '{trace_id}'.\n"
            "2. Output the Python patch strictly inside ```python ... ``` triple backticks.\n"
            "3. Clean 'df' directly: replace empty strings with NaN, convert types where necessary, and drop null rows.\n"
            "4. Do NOT include file read/write operations (no read_parquet/to_parquet)."
        ),
        expected_output="Executable Python Pandas snippet inside triple backticks ```python ... ```.",
        agent=de_agent
    )

    return Crew(
        agents=[sre_agent, de_agent],
        tasks=[task_diagnose, task_patch],
        process=Process.sequential,
        verbose=True
    )


def create_mle_crew() -> Crew:
    llm = get_crewai_llm()

    mle_agent = Agent(
        role="MLOps Release Engineer",
        goal="Evaluate model metrics, benchmark against the production Champion, and serialize ONNX artifacts.",
        backstory="You are a principal MLOps Engineer. You write standalone Python scripts that calculate validation metrics, benchmark Candidate vs. Champion in a Model Registry, and export verified ONNX graphs.",
        llm=llm,
        max_iter=5,
        max_execution_time=90,
        verbose=True
    )

    task_serialize = Task(
        description=(
            "Write a standalone Python script to evaluate candidate model metrics, compare against Champion model, and export ONNX.\n"
            "Requirements:\n"
            "1. Load dataset from '{data_path}' and model from 'models/core_prediction_model.joblib' or 'models/retrained_model.pt'.\n"
            "2. Calculate metrics: F1-Score, ROC-AUC, and average inference latency in milliseconds over 100 benchmark iterations.\n"
            "3. Read Champion metrics from 'models/model_registry.json'. (If missing, default Champion F1=0.80, ROC_AUC=0.82, Latency_ms=15.0).\n"
            "4. Champion vs Challenger Decision:\n"
            "   - If Candidate F1 > Champion F1 or (Candidate F1 == Champion F1 and Candidate Latency < Champion Latency):\n"
            "     Promote candidate status to 'PRODUCTION' (Champion).\n"
            "   - Otherwise, mark candidate status as 'CHALLENGER'.\n"
            "5. Export candidate to ONNX format ('models/pipeline_model.onnx') and verify ONNX Runtime predictions match base outputs within 1e-5.\n"
            "6. Save updated model metadata back into 'models/model_registry.json'.\n"
            "7. Output ONLY the Python script strictly enclosed inside ```python ... ``` triple backticks."
        ),
        expected_output="A standalone Python script enclosed in triple backticks ```python ... ```.",
        agent=mle_agent
    )

    return Crew(
        agents=[mle_agent],
        tasks=[task_serialize],
        process=Process.sequential,
        verbose=True
    )


async def run_de_crew_instrumented(inputs: dict) -> str:
    de_crew = create_de_crew()
    with logfire.span("crewai.de_crew", trace_id=inputs.get("trace_id")):
        res = await de_crew.kickoff_async(inputs=inputs)
        return str(res)


async def run_mle_crew_instrumented(inputs: dict) -> str:
    mle_crew = create_mle_crew()
    with logfire.span("crewai.mle_crew", data_path=inputs.get("data_path")):
        res = await mle_crew.kickoff_async(inputs=inputs)
        return getattr(res, "raw", str(res))
