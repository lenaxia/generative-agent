# Design Document for Multi-Agent Collaboration and Planning System

## 1. Introduction

The design revolves around creating a multi-agent collaboration system to handle complex tasks by dividing responsibilities across specialized agents. This architecture is built to enhance efficiency by routing specific tasks to the most appropriate "expert" agent, allowing each agent to operate within its domain while interacting through a shared communication system.

The planning system and the agent supervisor architecture are integral components of this design, enabling the orchestration of tasks across multiple agents. This document outlines the high-level design of these systems and how they interact with each other to accomplish large-scale tasks.


```go
generative-agent/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── web_search_agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── search.py
│   │   │   └── ...
│   │   └── models/
│   │       ├── __init__.py
│   │       └── ...
│   ├── coding_agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── code_writer.py
│   │   │   └── ...
│   │   └── models/
│   │       ├── __init__.py
│   │       └── ...
│   ├── math_agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── math_solver.py
│   │   │   └── ...
│   │   └── models/
│   │       ├── __init__.py
│   │       └── ...
│   └── ...
├── shared_tools/
│   ├── __init__.py
│   ├── s3.py
│   └── ...
├── supervisor/
│   ├── __init__.py
│   ├── supervisor.py
│   ├── task_graph.py
│   ├── request_manager.py
│   ├── heartbeat.py
│   ├── agent_manager.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_database.py
│   │   └── ...
│   ├── llm_provider/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       ├── json_schemas.py
│       ├── task_assignment.py
│       └── ...
├── schemas/
│   ├── task_assignment.json
│   ├── task_response.json
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── test_base_agent.py
│   │   ├── web_search_agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   └── ...
│   │   ├── csv_analysis_agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   └── ...
│   │   ├── math_agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   └── ...
│   │   ├── coding_agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   └── ...
│   │   ├── home_assistant_agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   └── ...
│   │   └── ...
│   ├── shared_tools/
│   │   ├── __init__.py
│   │   ├── test_base_tool.py
│   │   ├── test_s3.py
│   │   └── ...
│   ├── supervisor/
│   │   ├── __init__.py
│   │   ├── test_supervisor.py
│   │   ├── test_llm_provider.py
│   │   └── ...
│   └── ...
├── requirements.txt
├── setup.py
└── README.md
```


1. **agents/**
   * This directory contains the base agent class (`base_agent.py`) and subdirectories for specialized agents, such as web search, CSV analysis, math, coding, and home assistant agents.
   * Each specialized agent subdirectory (`web_search_agent/`, `csv_analysis_agent/`, `math_agent/`, `coding_agent/`, `home_assistant_agent/`) contains the following:
     * `agent.py`: The main file that defines the agent class for that specific agent type.
     * `tools/`: A directory for agent-specific tools, such as search engines, CSV processing utilities, math solvers, code execution environments, or home automation tools.
     * `models/`: A directory for agent-specific models, such as language models, machine learning models, or any other models required by the agent.
2. **shared_tools/**
   * This directory contains shared tools that can be used across multiple agents.
   * For example, `s3.py` is a shared tool for reading and writing data to an S3 bucket.
3. **supervisor/**
   * This directory contains the Supervisor component and its related modules.
   * `supervisor.py`: The main Supervisor class that oversees the entire workflow and agent coordination.
   * `task_graph.py`: A module for managing the task graph and its nodes and edges.
   * `request_manager.py`: A module for handling incoming requests and managing their lifecycle.
   * `agent_manager.py`: A module for managing the available agents and their assignments.
   * `memory/`: A subdirectory for modules related to the memory/knowledge base, such as the vector database (`vector_database.py`).
   * `utils/`: A subdirectory for utility modules, such as JSON schema definitions (`json_schemas.py`) and task assignment utilities (`task_assignment.py`).
4. **schemas/**
   * This directory contains JSON schema files for validating the structure of task assignments, task responses, and other JSON payloads used within the system.
5. **tests/**
   * This directory contains unit tests for the various components of the system.
   * It has subdirectories for testing the base agent (`test_base_agent.py`), specialized agents (`web_search_agent/test_agent.py`, `csv_analysis_agent/test_agent.py`, etc.), shared tools (`test_s3.py`), and the Supervisor (`test_supervisor.py`).
6. **requirements.txt**
   * This file lists the project's Python dependencies and their versions.
7. **[setup.py](http://setup.py)**
   * This is a Python script used for packaging and distributing the project.
8. **[README.md](http://README.md)**
   * This file contains documentation and instructions for the project.

## 2. Multi-Agent System Overview

The multi-agent system employs a "divide-and-conquer" approach where tasks are split among specialized agents. Each agent is equipped with a set of tools and is responsible for performing a distinct function. Complex operations are broken down and routed to different agents based on their expertise, which leads to efficient task completion.

Key characteristics:

* **Specialization**: Each agent focuses on a specific task or domain.
* **Collaboration**: Agents collaborate, passing partial results to one another.
* **Orchestration**: A supervisor coordinates the overall process and ensures smooth task transitions between agents.

Agents interact through a shared communication graph, allowing them to pass tasks and results to one another in a seamless workflow.

## 3. Planning System Architecture

The Planning System serves as the brain of the multi-agent system, determining the sequence in which tasks are executed and assigning them to the appropriate agents. The planning system operates on a task graph that represents the dependencies between different subtasks.

### 3.1 Task Graph

A task graph is a directed acyclic graph (DAG) where:

* Nodes represent individual agents or tools.
* Edges represent the flow of data and task dependencies.

The task graph starts with a high-level task provided by the user, which is decomposed into smaller subtasks, each of which is handled by a specific agent. Once an agent completes its task, the next agent in the graph is notified to begin its task.

### 3.2 Dynamic Task Routing

The system supports dynamic task routing, where the outcome of an agent's task may change the path that subsequent tasks follow. For example, based on the results from a research agent, a charting agent may either generate a graph or request additional data from another agent.

### 3.3 Conditional Task Execution

The system also supports conditional task execution, where agents can decide whether their output is final or requires further action. If the final result is reached, the graph is terminated early to save resources.

## 4. Supervisor Architecture

The Supervisor is a special agent tasked with managing the entire conversation and workflow between the agents. The supervisor's role is to:

* Delegate tasks to specific agents based on the task graph.
* Monitor agent progress and capture their outputs.
* Decide when to move to the next agent or terminate the workflow.

### 4.1 Supervisor as a Decision Maker

The supervisor is an LLM-driven agent that uses contextual data from the agents' outputs to determine the next steps in the process. It decides which agent to invoke next based on the current state of the conversation and task progress. The supervisor ensures that the right agent is activated at the right time.

### 4.2 Task Delegation

The supervisor delegates tasks by sending structured JSON queries to agents. It chooses the next agent based on the results received from previous agents and the overall goal of the workflow. This is done using a combination of predefined rules and LLM-based decision-making.

### 4.3 Heartbeats

The supervisor operates on a cycle of heartbeats and events. Heartbeats trigger the agent to perform actions without external stimuli, while events allow the agent to respond to external stimuli. During each cycle, the agent makes a decision on what functions to call based on tools available through the service locator. It is allowed to iterate on the results of its function call until it decides its turn is complete and its ready to respond to the event or has no more work left to do on its current task. In the case of a heartbeat, an initial call to the LLM backend is made with a prompt based on the current state of the system, existing tasks and their status, and previous actions that have been taken recently. An event is then generated and we merge into the same workflow as the event pathway.

Or if the event is a heartbeat, the Generative Agent may decide to use a "check home assistant status" service that would result in calls to a "weather" service, "time" service, "calendar" service to understand the schedule and timing of people in and out of the house to determine how it wants to configure the HVAC, window blinds and lights.

Or finally, if there is a capability that the generative agent cannot currently do, it could decide it wants to write a new module for itself, it would generate a plan and a design for what it wants, implement the code, then using SSH connect to a remote machine, and run and test the code it has written, once it is satisfied, it would then use the OS module to write the files to the modules folder for the module to be loaded into the store locator.

### 4.4 Parallel Task Management

The Supervisor is capable of handling multiple requests concurrently and distributing subtasks to available agents in parallel. It maintains a task queue or a pool of active tasks, assigning tasks to agents as they become available. Each agent has a unique identifier or handle, allowing the Supervisor to track their progress and manage their workloads effectively.

The Supervisor implements load balancing algorithms to distribute tasks evenly among agents based on their availability, skills, and current workloads.

### 4.5 Memory/Knowledge Base Integration

The Supervisor has access to a memory/knowledge base that stores relevant information, such as previous task results, research findings, or domain-specific knowledge. The memory/knowledge base is implemented using a vector database or a retrieval-augmented generation (RAG) system, which allows agents to retrieve relevant information based on their current task context.

Agents can read from and write to the memory/knowledge base, enabling them to build upon previous work and share knowledge with other agents. However, findings, results, and domain knowledge are collated with results and findings from the same tasks to reduce confusion and prevent intermingling of information from separate tasks.

### 4.6 Summary Agent

When a task is completed, a Summary Agent takes the findings and results of the task and summarizes them. This summary is stored in the memory/knowledge base, allowing the Supervisor and other agents to search the vector history and retrieve the abbreviated summary, even if they get a match on some of the more detailed historical entries for the task.

## 5. Agent Design

Each agent is a specialized worker that performs a specific function in the workflow. The design of each agent includes the following components:

### 5.1 Core Components

* **LLM Core**: Each agent is powered by a language model like GPT-4, which processes task instructions and generates outputs.
* **Tools**: Each agent has access to a unique set of tools, which it can invoke to perform its tasks. These tools may include search engines, Python REPL environments, or specialized APIs for domain-specific operations.

### 5.2 Tool Utilization

Agents use tools to gather information, process data, or perform computations. For example:

* A research agent might use a web search tool to gather data.
* A chart generation agent might use a Python REPL tool to generate plots or graphs.

### 5.3 Agent Parallelization

Certain agents may be able to process multiple tasks concurrently, especially if their tasks are independent or can be parallelized. These agents are designed to handle parallel task execution, either by spawning separate threads or processes, or by utilizing parallel computing libraries (e.g., Dask, Ray, or multiprocessing in Python).

## 6. Graph Construction and Execution

### 6.1 Agent Nodes

Each agent node in the task graph performs a single operation based on its expertise. When invoked, the agent processes the input it receives, performs the necessary task (such as running a search or generating a chart), and then passes the results to the next node in the graph.

### 6.2 Edge Logic

The edges between nodes determine how tasks flow through the graph. Based on the results of one agent, the graph's routing logic decides which agent should be invoked next. For example, if the research agent generates partial data, the system routes the task to the chart generation agent, unless additional research is required.

The routing mechanism ensures that agents collaborate effectively, passing tasks forward or re-routing tasks when additional steps are necessary.

## 7. Supervised Agent Execution

The Agent Supervisor oversees the execution of the entire workflow. It monitors agent outputs and determines when to move to the next stage in the task graph.

### 7.1 State Tracking

The supervisor tracks the overall state of the workflow, including:

* **Agent Outputs**: Captures intermediate results produced by each agent.
* **Task Status**: Tracks which agents have completed their tasks and which are still pending.

### 7.2 Task Completion

Once the agents reach a final result, the supervisor identifies the completion signal from the agents' outputs (in the form of a structured JSON response) and terminates the workflow. If no final result is found, it continues delegating tasks until the result is achieved.

## 8. Agent Supervisor Interaction with Task Graph

The Agent Supervisor interacts closely with the task graph, adjusting execution based on dynamic inputs and changing task conditions. It uses the outputs from agents to determine the flow of operations, adjusting the task graph as needed.

For example:

* If an agent outputs a JSON response indicating a final result, the supervisor concludes the task graph.
* If further action is required, the supervisor routes the next task to the appropriate agent or tool.

## 9. Error Handling and Recovery

The system includes robust error handling, where the supervisor can detect when an agent fails to complete its task. In such cases, the supervisor can either retry the task or re-route it to another agent capable of completing it.

## 10. Conclusion

This multi-agent collaboration system, powered by a supervisor and planning system, offers a scalable, efficient approach to handling complex tasks. By dynamically routing tasks and delegating them to specialized agents, the system can efficiently handle a variety of operations, from web research to data analysis and graph generation. The planning system ensures the proper flow of tasks, while the supervisor orchestrates agent interaction to achieve final results.

This document provides a detailed overview of the system and its components, including the supervisor and planning architecture, agent design, graph construction, and memory/knowledge base integrations

