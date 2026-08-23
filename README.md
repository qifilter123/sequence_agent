# Sequence Agent

Sequence Agent is a DAG-driven pipeline for sequence-pattern modeling, embedding analysis, clustering, and model diagnostics. The project is designed to support controlled model experimentation and automated diagnostic-agent reasoning without hard-coding the processing flow in orchestration scripts.

This file provides stable, human-maintained project context. Detailed and frequently changing implementation behavior belongs in the DAG definitions and source code.

## Processing Flow

The project follows this high-level flow:

1. Extract raw sequence fields.
2. Transform raw fields into model features.
3. Build and run the sequence model.
4. Generate predictions and calculate training losses.
5. Produce record-level embeddings.
6. Build HDBSCAN clusters and centroids from embeddings.
7. Evaluate new records against the built clusters.
8. Collect diagnostic evidence for analysis and experimentation.

Training and HDBSCAN inference share the same DAG-defined feature and model semantics.

## DAG Definitions

The processing pipeline is declared by the YAML files under `src/model_diagnostic/config/`:

| Definition | Responsibility |
| --- | --- |
| `input_extractor.yaml` | Extract raw sequence fields and expose the downstream raw-data contract. |
| `input_transformer.yaml` | Transform extracted fields and assemble ordered model input. |
| `model_structure.yaml` | Declare the model structure and model data flow. |
| `prediction_loss.yaml` | Declare prediction targets, masking, and training-loss computation. |
| `hdbscan_centroid_build.yaml` | Declare phase-1 embedding collection and centroid construction. |
| `hdbscan_evaluation.yaml` | Declare phase-2 cluster-based evaluation. |

These DAG definitions are the authoritative source for pipeline topology, operation selection, inputs, parameters, and data lineage. Python source provides the implementation of the operations referenced by the DAGs.

## Source Organization

The primary implementation is under `src/model_diagnostic/`:

| Path | Responsibility |
| --- | --- |
| `config/` | DAG definitions and generated diagnostic representations. |
| `dag/` | Generic DAG processing and operation registries. |
| `dag_ops/` | Feature, model, prediction/loss, and HDBSCAN DAG operations. |
| `diagnostic/` | Diagnostic probes, registries, runtime support, and model-structure diagnostics. |
| `model/` | Model-domain implementations and supporting components. |
| `model_trainer.py` | Training entry point. |
| `inference_hbscan.py` | HDBSCAN centroid-building and evaluation entry point. |
| `hdbscan_report_helper.py` | HDBSCAN reporting support. |
| `cfg_base.py` | Shared runtime configuration. |
| `requirements.md` | Python package dependencies required by the project runtime. |

## Diagnostic Agent and MCP Runtime

The MCP server is the controlled runtime interface between the Diagnostic Agent and this project.

Human developers run the trainer and HDBSCAN inference scripts inside the project's Python virtual environment. The Diagnostic Agent does not activate that virtual environment or invoke internal Python modules directly. It calls the capabilities exposed by the MCP API and relies on the MCP server runtime, which imports the required project dependencies.

Python package dependencies are declared in `requirements.md`. Dependency names and versions should be read from that file rather than duplicated in this README.

## Run the Trainer

Activate the project Python virtual environment, then run from the project root:

```bash
python src/model_diagnostic/model_trainer.py
```

The trainer loads the input, model, and prediction/loss DAGs, trains the sequence model, runs configured diagnostics, and saves the trained model artifact.

## Run HDBSCAN Inference and Evaluation

After a trained model is available, activate the same project Python virtual environment and run:

```bash
python src/model_diagnostic/inference_hbscan.py
```

The script performs two stages:

1. **Centroid build:** generate embeddings and build HDBSCAN clusters and centroid data.
2. **Evaluation:** evaluate records against the clusters built in the first stage and produce diagnostic reports.

## Context Collection Guidance

The Context Agent should use this file to determine the project scope and use the DAG definitions and source code to collect detailed context.

The primary context sources are:

- `README.md` for stable project purpose, boundaries, entry points, and high-level flow.
- `src/model_diagnostic/config/*.yaml` for authoritative processing topology and data lineage.
- `src/model_diagnostic/dag/` for DAG execution and registry behavior.
- `src/model_diagnostic/dag_ops/` for registered operation implementations.
- `src/model_diagnostic/model/` for model-domain implementations.
- `src/model_diagnostic/diagnostic/` for diagnostic instrumentation and runtime evidence collection.
- `model_trainer.py` and `inference_hbscan.py` for end-to-end orchestration and runtime entry points.
- `requirements.md` for project runtime dependencies.
- The MCP server implementation for the API boundary between the Diagnostic Agent and project runtime capabilities.

When building project context, connect relationships in this direction:

```text
DAG definition -> DAG node/op -> registry -> Python implementation -> model or pipeline component -> runtime entry point
```

For agent-triggered execution, also connect:

```text
Diagnostic Agent -> MCP API/tool -> MCP server handler -> project runtime entry point -> DAG definitions
```

Generated files such as `model_structure.diagnostic.json` are diagnostic evidence derived from the model DAG. They should retain provenance to their source DAG and code version, but they are not authoritative source definitions.

Do not treat virtual environments, Python caches, logs, model checkpoints, temporary outputs, or other generated artifacts as source-code context unless a diagnostic task explicitly requires them.

## Maintenance Boundary

Keep this README limited to information that is expected to remain stable across normal development. Do not manually duplicate details that can be collected from DAG YAML or source code, including:

- Current layer counts or enabled diagnostic-module counts.
- Model dimensions and hyperparameter values.
- Individual DAG node parameters.
- Current loss values, evaluation scores, or experiment results.
- Function signatures and low-level call relationships.

Those details should be discovered automatically so that the collected context remains synchronized with the implementation.
