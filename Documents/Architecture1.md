```mermaid
graph TD
    %% Styling definitions for visual hierarchy
    classDef infra fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff;
    classDef queue fill:#c05621,stroke:#dd6b20,stroke-width:2px,color:#fff;
    classDef agent fill:#2b6cb0,stroke:#3182ce,stroke-width:2px,color:#fff;
    classDef db fill:#553c9a,stroke:#6b46c1,stroke-width:2px,color:#fff;
    classDef llm fill:#b7791f,stroke:#d69e2e,stroke-width:2px,color:#fff;
    classDef decision fill:#276749,stroke:#2f855a,stroke-width:2px,color:#fff;

    %% Client and Ingestion Layer
    Client([Telemetry Source]) -->|HTTP POST JSON| API[FastAPI Uvicorn Gateway]
    API -->|Publish & Ack 202| NATS[(NATS JetStream Broker)]
    
    %% Processing and Persistence
    NATS -->|Async Pull| Worker[Python Consumer Worker]
    Worker --> Init[Initialize LangGraph State]
    Init <-->|Read/Write Checkpoints| PG[(PostgreSQL)]

    %% LangGraph Orchestration Subgraph
    subgraph "LangGraph State Machine"
        Init --> Auditor[Auditor Agent Node]
        
        Auditor --> Router{State Router}
        
        Router -->|state == NOMINAL| End([Transaction Finalized])
        Router -->|state == DRIFT_DETECTED| Janitor[Janitor Agent Node]
        
        Janitor -->|Apply Patch & Verify| Auditor
    end

    %% External LLM Factory
    Auditor <-->|LLM Factory Interface| Gemini[Gemini-Flash-Latest]
    Janitor <-->|LLM Factory Interface| Gemini

    %% Apply Classes
    class API,Worker,Init infra;
    class NATS queue;
    class Auditor,Janitor agent;
    class PG db;
    class Gemini llm;
    class Router decision;
