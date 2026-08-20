# Model Diagnostic Project

## Project Purpose

This project develops and diagnoses a PyTorch sequence model for learning transaction behavior from temporal, amount, and entity-change features.

The model is trained through multi-task next-step prediction, but training loss is an intermediate learning objective rather than the final measure of model usefulness.

The primary modeling goal is to learn sequence embeddings that:

- preserve meaningful transaction-behavior structure,
- distinguish anomalous behavior from legitimate behavioral variation,
- provide useful representations for downstream unsupervised anomaly detection.

The current downstream evaluation uses HDBSCAN-derived behavioral centroids and transaction-level centroid-spread scoring.

The diagnostic agent is therefore expected to reason across the full chain:

```text
implementation
    │
    ▼
training dynamics
    │
    ▼
learned sequence representation
    │
    ▼
embedding / cluster geometry
    │
    ▼
downstream anomaly quality
```

The diagnostic agent is expected to understand both:

- observed runtime diagnostics,
- the actual model/training implementation that produces those observations,
- downstream HDBSCAN evaluation of the learned representation.

Source code may be read whenever implementation details are relevant to a diagnostic question.

Do not infer implementation semantics from metric names when they can be verified directly from source.

---

## End Goal and Objective Hierarchy

Do not treat optimization health as the final model objective.

The model is trained using reconstruction / next-step prediction losses in order to learn useful behavioral representations.

Use the following conceptual hierarchy:

```text
Implementation correctness
        │
        ▼
Training health
        │
        ▼
Representation quality
        │
        ▼
Downstream anomaly quality
```

Training diagnostics such as:

- loss,
- activation scale,
- gradient magnitude,
- clipping behavior,
- parameter updates,

are evidence about the representation-learning process.

They are not, by themselves, proof that a model change improves the final model.

A change that produces numerically cleaner training but degrades downstream representation quality should not be considered an improvement.

In particular:

- lower training loss does not automatically imply better embeddings,
- smoother gradients do not automatically imply better embeddings,
- lower activation magnitude does not automatically imply better embeddings,
- reduced gradient clipping does not automatically imply better anomaly detection.

Treat unusual training behavior as something to investigate, not automatically something to remove or normalize.

Use downstream evaluation to determine whether a representation-level intervention is actually beneficial.

---

## Source of Truth

Use the following precedence when reasoning about the running experiment:

1. MCP runtime context for the active experiment configuration and model structure.
2. Runtime metrics for observed training behavior.
3. MCP HDBSCAN evaluation for observed downstream representation/anomaly behavior.
4. Source code for model, feature, loss, trainer, and evaluation semantics.
5. `references/metric-categories.md` for diagnostic metric definitions and baseline instrumentation scope.
6. Other project reference documents for stable project context and previously established findings.

The static documentation provides project context but does not override the active runtime configuration or observed runtime evidence.

Runtime training metrics describe observed optimization behavior.

HDBSCAN evaluation describes observed downstream behavior of the current learned representation.

Source code describes implementation semantics.

Keep these evidence types distinct.

Reading source code is allowed and encouraged when it can resolve uncertainty without requiring a new experiment.

Do not modify source code.

---

## Important Source Areas

The primary implementation areas are:

### `encoder_model_train.py`

- feature construction
- training loop
- loss computation
- multi-task loss components
- gradient clipping
- diagnostic metric recording

### `generic_model.py`

- `ForwardSeqEncoder`
- `ContinuousTimeEmbedding`
- `ResidualGRUBlock`
- prediction heads
- embedding extraction

### `generic_feature_util.py`

- temporal feature transforms
- amount transforms
- historical-ratio features
- switch encoding
- is-new target construction

### `cfg_base.py`

- fixed `CFG`
- experiment-tunable `CFG2`

### `encoder_eval_hbscan.py`

- downstream embedding extraction
- HDBSCAN centroid construction
- cluster diagnostics
- transaction-level centroid-spread scoring
- per-pattern anomaly evaluation

### `diagnostic_runtime.py`

- active in-memory model and optimizer
- bounded training execution
- training metric collection
- controlled CFG2 experiments
- HDBSCAN evaluation of the active model
- current/previous experiment comparison

Read these files, or other relevant project source, when necessary to verify a hypothesis or metric interpretation.

---

## Model Overview

The current model is `ForwardSeqEncoder`, a forward-only residual GRU sequence encoder with multi-task next-step prediction.

High-level flow:

```text
16 base transaction features
        │
        ├───────────────┐
        │               │
        │ raw dt        ▼
        │        ContinuousTimeEmbedding
        │               │
        │          time embedding
        │               │
        └──────┬────────┘
               │
               ▼
        concat [base, time]
               │
               ├──────────────► feature_corr MLP
               │                     │
               │                 combo features
               │                     │
               └──────────┬──────────┘
                          │
                          ▼
            concat [base, time, combo]
                          │
                          ▼
                    GRU stack
                          │
                          ▼
                  shared sequence state
                    │    │    │    │
                    ▼    ▼    ▼    ▼
                    v    sw   amt  is_new
                   head head  head  head
```

The prediction heads provide self-supervised / reconstruction-style learning signals.

The shared sequence representation is also used to produce embeddings for downstream anomaly analysis.

---

## Downstream HDBSCAN Evaluation

The current downstream evaluation operates on embeddings generated by the trained sequence encoder.

High-level flow:

```text
trained ForwardSeqEncoder
        │
        ▼
extract_embedding(...)
        │
        ▼
record-level embeddings
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               │
Phase 1: historical-style data          │
is_training=True                        │
        │                               │
        ▼                               │
HDBSCAN clustering                      │
        │                               │
        ▼                               │
behavioral cluster centroids            │
                                        │
                                        ▼
                              Phase 2: evaluation data
                              is_training=False
                                        │
                                        ▼
                              record embeddings
                                        │
                                        ▼
                         map views to nearest centroids
                                        │
                                        ▼
                        intra-transaction centroid spread
                                        │
                                        ▼
                              anomaly score / evaluation
```

The MCP runtime evaluates the current in-memory model directly.

Do not assume a checkpoint on disk represents the active MCP model.

### Phase 1 — Centroid Construction

The evaluator generates historical-style anchored records with:

```text
is_training=True
```

and uses their embeddings to fit HDBSCAN clusters.

These clusters define behavioral centroids used by downstream scoring.

### Phase 2 — Evaluation

Evaluation records are generated with:

```text
is_training=False
```

so the current transaction being evaluated is positioned according to evaluation semantics.

Each record view is mapped to its nearest learned centroid.

For each transaction, the current primary anomaly score is based on the maximum distance between different centroids reached by its record views:

```text
transaction
    │
    ├── anchor/view A → centroid X
    ├── anchor/view B → centroid X
    ├── anchor/view C → centroid Y
    └── anchor/view D → centroid Z

centroid spread
    =
max pairwise distance among {X, Y, Z}
```

Large centroid spread indicates that different views of the transaction occupy substantially different behavioral regions.

This is evidence that the learned embedding captures behavioral inconsistency or anomaly structure.

---

## Downstream Evaluation Metrics

Important global metrics include:

- ROC-AUC
- PR-AUC
- best F1
- precision
- recall / fraud detection rate
- benign false-positive rate
- confusion matrix

Do not optimize only one aggregate metric without examining its trade-offs.

Pattern-level evidence is also important.

Current anomaly patterns include behavior such as:

- `Stuffing_then_ATO`
- `Stuffing_rotating_proxy`
- `Stuffing_dumb_script`
- `Stuffing_card_tester`
- `Stuffing_low_and_slow`
- `ATO_sneaky`
- `ATO_blitz`
- `ATO_classic`

Important benign behavioral variants include:

- `Shopper`
- `Upgrader`
- `Traveler`
- normal behavior

A model change may improve one fraud pattern while degrading another or increasing false positives on legitimate behavioral variation.

Therefore inspect both:

```text
global anomaly quality
+
per-pattern behavior
```

before deciding whether an experiment is better.

---

## Interpretation Discipline

Always distinguish:

### Implementation Fact

Verified from project source.

Example:

```text
raw dt is passed to ContinuousTimeEmbedding
```

### Runtime Observation

Directly measured by diagnostics or downstream evaluation.

Example:

```text
time_emb input abs_max is very large
```

or:

```text
Traveler false-positive rate increased
```

### Derived Observation

Combines implementation and measured evidence without introducing an unverified mechanism.

Example:

```text
large raw dt values reach ContinuousTimeEmbedding during training
```

### Hypothesis

A possible explanation that still requires discriminating evidence.

Example:

```text
large raw dt may cause GRU gate saturation
```

Do not promote a hypothesis to a root cause solely because it is theoretically plausible.

---

## Diagnostic Abnormality vs Useful Representation

An unusual activation scale, gradient magnitude, clipping rate, or other training pattern is not automatically a defect.

Before recommending normalization, clipping changes, feature removal, or representation transformation, ask:

1. Is there evidence that this behavior is actually impairing training?
2. Is there evidence that it damages the learned representation?
3. Could the unusual signal contain useful behavioral information?
4. What experiment can distinguish these possibilities?

Representation-changing fixes require downstream evaluation.

For example:

```text
large feature scale
        │
        ▼
possible optimization concern
```

does not by itself justify:

```text
normalize feature
        │
        ▼
assume model improved
```

Instead:

```text
hypothesis
    │
    ▼
controlled intervention
    │
    ├── training diagnostics
    │
    └── HDBSCAN evaluation
            │
            ▼
      accept / reject
```

---

## Experiment Methodology

Prefer experiments that isolate causal factors.

Change the minimum number of variables required to test a hypothesis.

If a proposed intervention changes two independent mechanisms, separate them when practical.

For example, for two changes A and B:

```text
baseline      A0 B0
A only        A1 B0
B only        A0 B1
combined      A1 B1
```

This helps distinguish:

- the effect of A,
- the effect of B,
- interaction between A and B.

Do not attribute the result of a multi-change experiment to one component without discriminating evidence.

For CFG2 experiments, prefer matched training windows and use current/previous comparison tools.

For representation-changing source recommendations that the agent cannot implement itself, recommend the smallest useful experiment and explain what evidence would confirm or reject the hypothesis.

---

## Training and Downstream Evidence Together

When HDBSCAN evaluation is available, reason jointly over training and downstream evidence.

Examples:

```text
training improves
HDBSCAN improves
    → strong evidence of beneficial change
```

```text
training improves
HDBSCAN degrades
    → optimization improved but representation worsened
```

```text
training worsens slightly
HDBSCAN improves materially
    → change may still be beneficial for the actual modeling goal
```

```text
training unchanged
HDBSCAN changes materially
    → investigate representation geometry rather than optimizer behavior
```

Do not allow upstream diagnostic aesthetics to override stronger downstream evidence.

---

## Autonomous Diagnostic Goal

The diagnostic agent should use its available tools to iteratively reduce uncertainty:

```text
initialize baseline
        │
        ▼
inspect context / source
        │
        ▼
collect training evidence
        │
        ▼
evaluate HDBSCAN when useful
        │
        ▼
form competing hypotheses
        │
        ▼
choose smallest discriminating experiment
        │
        ▼
train controlled experiment
        │
        ▼
compare training behavior
        │
        ▼
compare downstream behavior
        │
        ▼
accept / reject hypothesis
        │
        ▼
repeat only while useful uncertainty remains
```

The goal is not to maximize the number of experiments.

Stop when available evidence is sufficient for an actionable conclusion or when the available tools cannot further distinguish the remaining hypotheses.