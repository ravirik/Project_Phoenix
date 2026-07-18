# Project Phoenix 🦅

A self-healing, multi-agent MLOps framework designed to automate the Machine Learning Engineering (MLE) lifecycle. Phoenix dynamically detects data drift, synthesizes remediation patches via Agentic RAG, and unblocks downstream model retraining pipelines—all orchestrated within a cloud-native Kubernetes environment.

Built as a capstone dissertation project demonstrating enterprise-grade AI infrastructure, Hexagonal Architecture, and scalable container orchestration.

## 🏗️ Core Architecture & K8s Deployment

Project Phoenix utilizes a fully decoupled, event-driven microservices architecture, designed to be deployed and scaled on Kubernetes:

1. **Ingestion Gateway (FastAPI / K8s Ingress):** A high-throughput REST endpoint acting as a fire-and-forget producer. It validates incoming pipeline telemetry and publishes it to the event bus.
2. **Event Broker (NATS JetStream):** Provides durable message queuing (`telemetry.raw.*`). Ensures no telemetry is lost during traffic spikes and decouples ingestion from heavy compute.
3. **AI Pull-Consumer (Python Worker Pods):** Horizontally scalable worker pods that pull events from NATS and trigger the LangGraph state machine. K8s HPA (Horizontal Pod Autoscaler) scales these consumers based on queue depth.
4. **Agentic State Machine (LangGraph):**
    * **Auditor Agent:** Analyzes telemetry to detect data drift or pipeline degradation before it corrupts model training.
    * **Janitor Agent:** Utilizes Gemini 3.5 Flash and tool-calling to embed anomaly signatures, query Qdrant for historical MLE playbooks, and synthesize remediation code.
    * **Executor Node:** A secure execution environment that applies the patch to the corrupted dataset, validating data integrity.
5. **Downstream MLE Trigger:** Upon successful remediation, the system signals the CI/CD pipeline to resume automated model feature-engineering and retraining.
6. **Vector Memory (Qdrant):** Stores verified historical incidents and their corresponding patches, deployed as a stateful set.

## 🚀 Current State & Capabilities

- [x] **Event-Driven Architecture:** Complete decoupling of HTTP ingestion from the AI processing layer via NATS.
- [x] **Agentic RAG:** Integration of Qdrant vector search for dynamic playbook retrieval.
- [x] **LLM Tool-Calling:** Strict Pydantic schema enforcement to prevent LLM parameter hallucination.
- [x] **Dynamic Execution:** Automated, on-the-fly execution of Python patches to heal pipeline data.

## 🗺️ Roadmap (Next Phases)

* **Kubernetes Orchestration:** Dockerize the Gateway and Worker, write Kubernetes manifests (Deployments, Services, StatefulSets), and deploy to a local cluster (e.g., Minikube/Kind).
* **MLE Lifecycle Integration:** Connect the Executor's success state to a mock ML training pipeline (e.g., triggering a PyTorch training job).
* **Secure Execution Sandbox:** Isolate the Executor node to prevent destructive code execution during the automated patching phase.
* **Continuous Memory Updates:** Implement a feedback loop that embeds novel remediation patches back into Qdrant for continuous learning.

## 🛠️ Tech Stack

* **Infrastructure:** Kubernetes, Docker, NATS JetStream
* **AI & LLM:** Google Gemini 3.5 Flash (Enterprise ADC), LangGraph, LangChain
* **Vector DB:** Qdrant
* **MLE / Data:** PyTorch, Pandas, PyArrow, Parquet
* **Backend:** FastAPI, Python (asyncio)