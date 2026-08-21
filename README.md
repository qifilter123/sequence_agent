# Sequence Agent

Sequence Agent is an experimental framework for **generic sequence-pattern modeling and AI-assisted model diagnostics**.

It combines configuration-driven DAG pipelines, PyTorch sequence models, learned embeddings, HDBSCAN clustering, and MCP-backed diagnostic agents to support controlled model experimentation.

## Architecture

```text
Raw Sequence Data
        |
        v
Input / Feature DAG
        |
        v
Sequence Model
        |
        v
Prediction / Loss
        |
        v
Embeddings
        |
        v
HDBSCAN
        |
        v
Evaluation
```

The pipeline is designed to be domain-neutral and reusable across different types of time-ordered event data.

## Agent-Assisted Diagnostics

The project includes a multi-agent diagnostic workflow backed by MCP tools.

Agents can:

* inspect model configuration and training behavior
* analyze losses, gradients, and representation quality
* run bounded experiments over predefined parameters
* compare downstream embedding and clustering results
* recommend targeted model or training changes

The goal is to reduce repetitive manual model troubleshooting while keeping experiments controlled and evidence-driven.

```text
Evidence
   |
   v
Diagnostic Agent
   |
   v
MCP Tools
   |
   v
Bounded Experiment
   |
   v
Training / Evaluation
   |
   v
Comparison
   |
   +----> next hypothesis
```

## DAG-Based Design

Model processing is represented through configurable DAGs rather than being fully hardcoded into training scripts.

The DAGs make dependencies between feature processing, model structure, prediction/loss, and downstream evaluation more explicit for both developers and diagnostic agents.

Operations distinguish between:

* `inputs` — pipeline values
* `cfg_params` — runtime configuration
* `params` — literal operation parameters

## Current Capabilities

* generic sequence feature processing
* PyTorch sequence modeling
* configurable DAG execution
* learned sequence embeddings
* HDBSCAN-based pattern discovery
* model diagnostics through MCP
* multi-agent diagnostic workflows
* controlled model experiments
* downstream evaluation of representation quality

## Scope

This repository is a **generic experimental project** for sequence modeling and model diagnostics.

It is not tied to a specific production system or business domain and is designed to work with synthetic or domain-neutral sequence data.

## Technologies

Python · PyTorch · HDBSCAN · YAML DAGs · MCP · Claude

## Goal

The project explores how structured model pipelines and tool-enabled AI agents can make sequence-model development more **repeatable, diagnosable, and easier to experiment with**.