# Model Diagnostic Evidence and Metric Categories

## Purpose

This reference documents the metric families and downstream evaluation evidence available to the model-diagnostic agents.

The current evidence scope is determined by:

1. the training runtime, which records training-level metrics,
2. `model_structure.diagnostic.json`, which enables module-level probes,
3. the MCP HDBSCAN evaluator, which evaluates downstream behavior of the learned representation produced by the active in-memory model.

This file defines:

- metric names and semantics,
- baseline diagnostic categories,
- baseline module scope and collection cadence,
- HDBSCAN evaluation output semantics,
- current/previous experiment evidence semantics,
- interpretation boundaries between training diagnostics and downstream evaluation.

It does not define the effective CFG2 values or model structure of every controlled experiment.

For an active experiment, MCP runtime context is authoritative.

Training diagnostics and HDBSCAN evaluation answer different questions:

```text
training metrics
    → optimization and representation-learning behavior

HDBSCAN evaluation
    → downstream usefulness of the learned representation
       under the current project evaluation setup
```

Do not use one evidence family as a substitute for the other.

---

## 1. Default Baseline Model Snapshot

The default baseline currently uses:

- class: `ForwardSeqEncoder`
- total parameters: `174567`
- trainable parameters: `174567`
- hidden dimension: `128`
- base feature dimension: `16`
- time embedding dimension: `16`
- feature-correlation output dimension: `16`
- switch classes: `32`
- is-new classes: `5`

These values describe the default baseline and must not be assumed to remain identical after a controlled CFG2 experiment.

Use current runtime context for the effective configuration and model structure of the active experiment.

Use previous-experiment context when structural comparison with the replaced experiment is required.

`structure_id` is informational only and must not be treated by itself as proof of structural equivalence, incompatibility, or correctness.

---

## 2. Baseline Instrumented Module Scope

The baseline diagnostic profile enables activation and parameter-gradient diagnostics for these logical modules:

| Module | Role / Baseline Static Information | Activation | Gradient |
|---|---|---:|---:|
| `time_emb` | `ContinuousTimeEmbedding`, baseline output dim 16 | enabled | enabled |
| `feature_corr` | logical feature-correlation MLP | enabled | enabled |
| `fwd_stack.0` | first `ResidualGRUBlock`, baseline residual disabled | enabled | enabled |
| `fwd_stack.1` | second `ResidualGRUBlock`, baseline residual enabled | enabled | enabled |
| `v_head` | baseline Linear 128 → 1 | enabled | enabled |
| `sw_head` | baseline Linear 128 → 32 | enabled | enabled |
| `amt_head` | baseline Linear 128 → 1 | enabled | enabled |
| `is_new_head` | baseline Linear 128 → 5 | enabled | enabled |

All other module records are disabled by the baseline diagnostic profile.

Examples include:

- `feature_corr.0` through `feature_corr.4`
- `fwd_stack.0.gru`
- `fwd_stack.0.ln`
- `fwd_stack.0.drop`
- `fwd_stack.1.gru`
- `fwd_stack.1.ln`
- `fwd_stack.1.drop`
- root module
- `fwd_stack` container

Static runtime model context may describe modules that are not instrumented.

Do not assume runtime diagnostic metrics exist for a module merely because the module exists in the model.

The diagnostic configuration path is outside the controlled CFG2 experiment boundary and is not modified by agent experiments.

---

## 3. Diagnostic Categories and Cadence

For every module enabled by the baseline diagnostic profile:

- `enabled = true`
- `activation = true`
- `gradient = true`
- `parameter = false`
- `update = false`
- `backward_flow = false`

The diagnostic JSON does not override collection cadence, so current `StageProbeConfig` defaults apply:

| Category | Collection Cadence |
|---|---:|
| Activation | every 50 diagnostic steps |
| Parameter gradient | every 50 diagnostic steps |

Activation input and output capture are enabled.

Step `0` is a collection step.

Module-level metrics therefore do not necessarily have a record on every training step.

When a controlled experiment restarts the session, it receives a fresh probe set and metric registry.

HDBSCAN evaluation does not use this cadence and is not stored as ordinary per-step registry history.

---

## 4. Training-Level Metrics

The diagnostic runtime records training-level metrics under stage `training`.

Most of these are recorded on every attempted training iteration while the corresponding code path is reached.

Use `list_metrics` to determine which metrics actually exist in the active session.

### 4.1 `loss.total`

Loss used for backward propagation.

Current baseline semantics:

```text
lambda_recon * loss.recon
```

If a controlled tunable changes an effective loss-related configuration, use runtime context and source code to determine the active semantics.

### 4.2 `loss.recon`

Current multi-task next-step reconstruction objective.

It combines the weighted task losses for:

- time-related prediction,
- switch-pattern prediction,
- amount-related prediction,
- is-new prediction,

normalized by valid sequence tokens according to the training implementation.

### 4.3 Component Loss Metrics

When `compute_nextstep_recon_loss` returns the current structured loss-detail mapping, the runtime records:

- `loss.v`
- `loss.sw`
- `loss.amt`
- `loss.is_new`

These values are normalized task-level losses before the corresponding task-specific lambda is applied inside the reconstruction objective.

The diagnostic runtime remains backward-compatible with the older trainer contract that returned only:

```text
(total_loss, sw_ce_float)
```

Therefore the component metrics are conditional on the active trainer return contract.

Do not assume that `loss.v`, `loss.sw`, `loss.amt`, or `loss.is_new` exist until runtime metric discovery confirms them.

#### `loss.v`

Normalized time-related next-step prediction loss.

#### `loss.sw`

Normalized switch-pattern loss used by the reconstruction objective before its task-specific lambda.

Under the current implementation, switch smoothing/weighting may be an identity operation, which can make this numerically equal to `loss.sw_ce`.

Do not assume that equality as a permanent semantic guarantee; verify the active implementation when the distinction matters.

#### `loss.amt`

Normalized amount-related next-step prediction loss.

#### `loss.is_new`

Normalized multi-label is-new prediction loss.

### 4.4 `loss.sw_ce`

Raw switch-class cross-entropy averaged over valid tokens.

This remains available under both the older and newer loss-return contracts.

It is useful for distinguishing raw switch classification cross-entropy from any current or future transformation used by `loss.sw`.

### 4.5 `loss.nonfinite`

Boolean indicating whether `loss.total` is non-finite.

Current semantics:

```text
not isfinite(loss.total)
```

This is a numerical-integrity signal, not a quality metric.

### 4.6 `optimizer.lr`

Learning rate from the first optimizer parameter group.

The value may differ between controlled experiments if learning rate is exposed as an allowed CFG2 tunable.

### 4.7 `grad.global.l2_pre_clip`

Global gradient norm returned by:

```text
torch.nn.utils.clip_grad_norm_
```

This value represents the global norm before clipping.

Module-level parameter-gradient diagnostics are collected after `loss.backward()` and before clipping.

### 4.8 `grad.global.nonfinite`

Boolean indicating whether the measured pre-clipping global gradient norm is non-finite.

This is a numerical-integrity signal.

### 4.9 `grad.clip.max_norm`

Configured clipping threshold used for the active training step.

The default baseline value may be `1.0`, but the active value must be taken from runtime evidence because `clip_val` may be tunable.

This metric is the configured threshold, not a measured post-clipping gradient norm.

### 4.10 `grad.clip.triggered`

Boolean indicating whether the finite pre-clipping global norm exceeded the configured clipping threshold.

Current semantics:

```text
isfinite(grad.global.l2_pre_clip)
and
grad.global.l2_pre_clip > grad.clip.max_norm
```

### 4.11 `optimizer.step_applied`

Boolean indicating whether an optimizer update was applied for the diagnostic step.

A normal successful training step records:

```text
True
```

If an exception occurs before the optimizer step is applied, the runtime attempts to record:

```text
False
```

This metric is important when interpreting partial or failed training segments.

A metric record from a failed step must not be interpreted as evidence that model parameters were updated unless `optimizer.step_applied` confirms it.

---

## 5. Training Segment Summaries

`train_segment` also returns compact segment-level summaries.

These summaries are tool results rather than ordinary `DiagnosticRegistry` metric histories.

Typical summary evidence includes:

- segment status,
- segment id,
- start step,
- end step,
- requested/completed steps,
- elapsed time,
- summary statistics for total loss,
- reconstruction loss,
- switch cross-entropy,
- available component losses,
- pre-clipping global gradient norm,
- clipping frequency and threshold,
- structured error information for failed segments.

Use detailed metric queries when temporal shape or individual-step evidence matters.

Use segment summaries for compact orientation and bounded experiment comparison.

---

## 6. Activation Metrics

For each enabled logical module, activation diagnostics record input and output statistics.

### Input Metrics

- `activation.input.shape`
- `activation.input.mean`
- `activation.input.std`
- `activation.input.rms`
- `activation.input.abs_max`
- `activation.input.zero_frac`
- `activation.input.nonfinite_frac`

### Output Metrics

- `activation.output.shape`
- `activation.output.mean`
- `activation.output.std`
- `activation.output.rms`
- `activation.output.abs_max`
- `activation.output.zero_frac`
- `activation.output.nonfinite_frac`

### Semantics

`shape`
: Actual runtime tensor shape captured by the forward hook.

`mean`
: Mean of finite tensor values.

`std`
: Standard deviation of finite tensor values.

`rms`
: Root-mean-square magnitude of finite tensor values.

`abs_max`
: Maximum absolute finite value.

`zero_frac`
: Fraction of all elements that are finite and exactly zero.

`nonfinite_frac`
: Fraction of elements that are NaN or Inf.

Non-finite elements are excluded from finite-value reductions.

Activation metrics can describe:

- runtime dimensional behavior,
- representation scale,
- dispersion,
- sparsity,
- numerical integrity,
- stage-to-stage differences,
- changes over training time,
- changes between controlled experiments.

Activation magnitude alone does not establish prediction quality, embedding quality, calibration, or anomaly-detection quality.

---

## 7. Parameter-Gradient Metrics

For each enabled logical module:

- `grad.param.l2`
- `grad.param.rms`
- `grad.param.abs_mean`
- `grad.param.abs_max`
- `grad.param.coverage`
- `grad.param.nonfinite_frac`

These metrics are collected after `loss.backward()` and before gradient clipping.

### Semantics

`grad.param.l2`
: Total L2 norm across collected trainable parameter gradients.

`grad.param.rms`
: RMS gradient magnitude across collected gradient elements.

`grad.param.abs_mean`
: Mean absolute gradient magnitude.

`grad.param.abs_max`
: Maximum absolute finite gradient value.

`grad.param.coverage`
: Gradient parameter elements divided by trainable parameter elements.

`grad.param.nonfinite_frac`
: Fraction of gradient elements that are NaN or Inf.

Raw L2 magnitude depends on module size.

For comparisons of per-parameter gradient scale across differently sized modules, `grad.param.rms` or `grad.param.abs_mean` is generally more directly comparable.

Parameter-gradient magnitude describes gradient signal; it does not measure optimizer update magnitude unless update diagnostics are separately enabled.

---

## 8. Available Evidence Categories

The current diagnostic system provides four major evidence categories:

1. training dynamics,
2. activation / intermediate representation behavior,
3. parameter gradients,
4. downstream HDBSCAN evaluation.

The first three primarily describe how the model is training.

The fourth describes whether the resulting learned representation is useful under the project's current downstream anomaly-evaluation method.

Possible relationships include:

- task loss versus global gradient behavior,
- clipping behavior versus training progress,
- activation behavior versus stage-level gradients,
- one logical stage versus another,
- the same stage across different training periods,
- training behavior across current and previous experiments,
- HDBSCAN quality across current and previous experiments,
- training changes versus downstream changes.

These are possible comparisons, not mandatory checks.

An improvement in an upstream diagnostic does not automatically constitute a model improvement.

For model-selection decisions, downstream HDBSCAN evidence has greater decision relevance than optimization aesthetics alone, provided the training process remains valid and numerically stable.

---

## 9. Downstream HDBSCAN Evaluation

The MCP runtime can evaluate the active in-memory model using the project's HDBSCAN centroid-spread evaluation.

Use:

```text
evaluate_hdbscan
```

for the active model.

Use:

```text
get_previous_evaluation
```

for the replaced experiment when a valid previous evaluation was captured before restart.

HDBSCAN evaluation is experiment-level evidence.

It is not stored as ordinary per-step `DiagnosticRegistry` history.

### 9.1 Model Identity and Freshness

Every successful evaluation is associated with the exact active model state through fields including:

- `session_id`
- `session_step`
- `model_step`
- `evaluated_at`
- `elapsed_seconds`
- `cfg2`

The runtime evaluates the active in-memory model directly.

Do not assume that a checkpoint on disk represents the model evaluated by MCP.

A successful optimizer update invalidates the previous current-model evaluation because the model weights have changed.

Therefore an HDBSCAN result must correspond to the intended final `model_step` before it is used for model selection or current/previous comparison.

### 9.2 Evaluation Identity

A completed result currently identifies the evaluation as:

```text
hdbscan_centroid_spread_v4
```

and includes the effective HDBSCAN evaluation configuration under:

```text
config
```

A failed evaluation returns:

- `status = failed`
- session/model identity,
- timing/config context,
- structured error type and message.

A failed HDBSCAN run is not valid downstream model-quality evidence.

### 9.3 Evaluation Flow

The current evaluation has two main phases.

#### Phase 1 — Centroid Construction

Historical-style anchored records are generated with:

```text
is_training=True
```

The active model produces record-level embeddings.

HDBSCAN is fit on those embeddings to build behavioral clusters and centroids.

The HDBSCAN clustering fit itself is unsupervised with respect to fraud labels.

After clustering, generated scenario/fraud labels are used for diagnostic interpretation such as:

- cluster fraud proportion,
- `FRAUD` versus `NORMAL` assignment,
- semantic cluster summaries.

Do not describe those post-fit diagnostic assignments as label-free.

#### Phase 2 — Evaluation

Evaluation-style records are generated with:

```text
is_training=False
```

The active model again produces record-level embeddings.

Views are mapped to learned centroids.

The current primary transaction-level anomaly score is intra-transaction centroid spread: the maximum pairwise distance among distinct centroids reached by the transaction's views.

Conceptually:

```text
transaction
    ├── view A → centroid X
    ├── view B → centroid X
    ├── view C → centroid Y
    └── view D → centroid Z

centroid spread
    =
max pairwise distance among {X, Y, Z}
```

Large centroid spread indicates that different views of a transaction occupy substantially different learned behavioral regions.

This is evidence about the geometry of the learned representation under the current evaluation design.

---

## 10. HDBSCAN Global Centroid-Spread Metrics

The main downstream score family is returned under:

```text
centroid_spread
```

### `roc_auc`

ROC-AUC of transaction-level centroid spread against the generated fraud labels.

Higher is better as a ranking metric, subject to the limitations of the current generated evaluation setup.

### `pr_auc`

Precision-recall AUC of transaction-level centroid spread against generated fraud labels.

Because fraud is the positive class and may be less frequent than benign behavior, PR-AUC is often especially informative for fraud-ranking quality.

### `best_f1`

Best F1 found over the evaluated centroid-spread thresholds.

This is an evaluation-selected operating point, not a fixed production threshold.

### `threshold`

Centroid-spread threshold associated with `best_f1`.

Do not compare threshold values alone as a model-quality metric; representation geometry can change the scale of the score.

### `precision`

Precision at the best-F1 threshold.

### `recall`

Recall at the best-F1 threshold.

### `accuracy`

Accuracy at the best-F1 threshold.

### `fraud_detection_rate`

Transaction-level fraud detection rate at the best-F1 threshold.

Under the current implementation this corresponds to:

```text
TP / (TP + FN)
```

and is numerically equivalent to positive-class recall.

### `benign_fp_rate`

Benign false-positive rate at the best-F1 threshold.

Current semantics:

```text
FP / (FP + TN)
```

Lower is generally better, but it must be interpreted together with fraud recall and other downstream metrics.

### `confusion`

Structured counts:

- `tp`
- `fp`
- `fn`
- `tn`

at the best-F1 threshold.

### Dataset Counts

The centroid-spread metric block also reports:

- `transactions_total`
- `fraud_transactions`
- `benign_transactions`

These counts help verify that comparisons were run on comparable evaluation populations.

---

## 11. HDBSCAN Pattern-Level Metrics

Pattern-level centroid-spread evidence is returned under:

```text
per_pattern
```

Each item currently contains:

- `scenario`
- `n`
- `is_fraud`
- `role`
- `mean_centroid_spread`
- `flag_rate`
- `suppressed_rate`

### `scenario`

Generated scenario name.

### `n`

Number of evaluated transactions for the scenario.

### `is_fraud`

Whether the generator classifies the scenario as anomalous/fraud.

### `role`

Diagnostic interpretation:

- `detect` for fraud/anomalous patterns,
- `FP-rate` for benign patterns.

### `mean_centroid_spread`

Mean transaction-level centroid-spread score for the scenario.

This is useful for understanding how the representation geometry separates behavioral patterns.

It is not itself a thresholded detection rate.

### `flag_rate`

Fraction of transactions in the scenario whose centroid spread is at or above the best-F1 threshold selected for the overall evaluation.

For fraud patterns, this acts as a scenario detection rate.

For benign patterns, it acts as a scenario false-positive rate.

### `suppressed_rate`

Fraction of scenario transactions suppressed by the evaluator's benign-cluster suppression logic, if that logic is active.

Under the current reviewed implementation, suppression may be inactive, in which case this can remain zero.

Do not infer active suppression solely from the presence of this field.

### Pattern-Level Interpretation

Pattern-level evidence is important because aggregate metrics can hide materially different behavioral trade-offs.

A model can:

- improve one fraud pattern,
- degrade another fraud pattern,
- improve aggregate ranking,
- increase false positives on a legitimate behavior,
- or produce the reverse combination.

Important benign pattern regressions should not be ignored merely because an aggregate score improves.

---

## 12. HDBSCAN Alignment Evidence

The evaluator also returns record-to-fraud-centroid alignment diagnostics under:

```text
alignment
```

This is analysis evidence about cluster geometry and is not the primary centroid-spread model-selection score.

When available, fields include:

- `available`
- `pure_cluster_count`
- `fraud_centroid_count`
- `transactions_total`
- `transactions_aligned`
- `fraud_aligned`
- `benign_aligned`
- `per_scenario`

If no FRAUD centroid passes the configured purity gate:

```text
available = false
```

with a reason.

### Per-Scenario Alignment

Each alignment scenario entry can contain:

- `scenario`
- `n`
- `is_fraud`
- `role`
- `alignment_rate`

`alignment_rate` is the fraction of transactions in that scenario with at least one evaluated record aligned within the radius of a selected FRAUD centroid.

Use alignment evidence to help explain representation/cluster behavior.

Do not substitute it for the transaction-level centroid-spread metrics unless the evaluation objective is explicitly changed.

---

## 13. HDBSCAN Cluster-Build Evidence

Compact cluster diagnostics are returned under:

```text
cluster_build
```

### Aggregate Fields

- `cluster_count`
- `fraud_cluster_count`
- `normal_cluster_count`

### Noise

The `noise` block contains:

- `count`
- `n_anomaly`
- `top_scenarios`

These describe HDBSCAN points assigned to noise during centroid construction.

### Per-Cluster Summaries

Each serialized cluster can contain:

- `label`
- `size`
- `n_anomaly`
- `n_normal`
- `p_fraud`
- `purity`
- `assigned`
- `semantic_type`
- `anchor_purity`
- `anchor_dominant`
- `anchor_mix`
- `dominant_scenarios`
- `max_align_distance`
- `median_align_distance`

Centroid vectors themselves are intentionally omitted from MCP transport.

Cluster diagnostics are explanatory evidence about learned embedding geometry.

They are not by themselves the final model-selection objective.

---

## 14. HDBSCAN Evaluation Dataset Evidence

The evaluator returns compact evaluation-population information under:

```text
evaluation_dataset
```

Current fields include:

- `transactions`
- `record_count_with_len_gate`
- `record_count_all`
- `fraud_transactions`
- `benign_transactions`

Use these fields to verify that compared experiments were evaluated under comparable dataset-generation conditions.

The HDBSCAN evaluation also returns its effective evaluator configuration under:

```text
config
```

When comparison validity matters, inspect material configuration differences rather than assuming two HDBSCAN runs are comparable solely because they used the same model architecture.

---

## 15. Registry and Query Semantics

The current runtime `DiagnosticRegistry` stores training/module metrics by:

- `stage`
- `metric_name`

Each record carries:

- `timestamp`
- `step`
- `epoch`
- `phase`
- `tags`

Runtime metric discovery determines what has actually been recorded.

A metric documented here may not yet appear if:

- its first scheduled collection event has not occurred,
- the active trainer uses an older compatible contract that does not emit that metric,
- execution failed before the metric's recording point.

Module-level metrics must be interpreted according to their collection cadence rather than assumed to exist on every training step.

Current and previous experiments have separate training metric registries/snapshots.

Previous-experiment metrics become available only after a successful controlled restart has replaced an active session.

HDBSCAN evaluations are maintained separately from the `DiagnosticRegistry`.

Do not query HDBSCAN evidence through ordinary training metric queries.

Use the HDBSCAN evaluation tools and experiment-comparison output instead.

---

## 16. Current and Previous Experiment Evidence

A successful controlled restart can expose evidence from the immediately replaced experiment.

The runtime can provide:

### Training / Configuration Evidence

- previous CFG2,
- previous training metric snapshot,
- previous model structure.

### Downstream Evidence

- previous HDBSCAN evaluation, when the replaced model was successfully evaluated at its final model step before restart.

A previous HDBSCAN evaluation is not automatically available merely because previous training metrics exist.

For valid downstream A/B comparison, both models must have fresh evaluations corresponding to the compared final model states.

The runtime's compact experiment comparison can expose HDBSCAN deltas when both current and previous evaluations are valid.

Current global metric deltas are defined for:

- `roc_auc`
- `pr_auc`
- `best_f1`
- `precision`
- `recall`
- `accuracy`
- `fraud_detection_rate`
- `benign_fp_rate`

Pattern comparison currently provides deltas for:

- `mean_centroid_spread`
- `flag_rate`

along with scenario identity and previous/current sample counts.

Threshold and confusion-matrix values remain available in the full evaluation result but are not currently part of the compact delta block.

---

## 17. Controlled Experiment Evidence

Controlled CFG2 experiments can change training behavior and the learned representation while retaining the fixed source-code and diagnostic-policy boundaries.

The runtime-defined tunable allowlist determines which configuration values may be changed.

This reference does not define that allowlist.

For experiment interpretation:

- current runtime context defines the active effective CFG2,
- previous-experiment context defines the replaced experiment,
- training metric queries provide detailed optimization evidence,
- HDBSCAN evaluation provides downstream representation evidence,
- compact experiment comparison may provide matched training windows and downstream metric deltas.

For current/previous controlled experiments:

1. inspect which CFG2 tunables changed,
2. compare training evidence over matched or comparable step windows,
3. verify HDBSCAN evaluations correspond to the intended model steps,
4. compare global HDBSCAN metrics,
5. inspect material pattern-level improvements and regressions,
6. distinguish optimization changes from downstream representation changes.

Changing a tunable and observing a training-diagnostic difference can provide evidence about training behavior.

It does not by itself establish a better model for the current downstream objective.

When the end goal is model selection, HDBSCAN evaluation is the currently available downstream evidence for whether the learned representation improved under this project's generated anomaly-evaluation setup.

---

## 18. Evidence Interpretation Hierarchy

Different evidence families answer different questions.

| Evidence | Primary Question |
|---|---|
| Runtime context / source | What is actually configured and implemented? |
| Training loss | Is the learning objective progressing, and which tasks dominate it? |
| Activation metrics | What scales and intermediate representations are flowing through the model? |
| Gradient metrics | Where and how strongly is learning signal flowing? |
| Optimizer-step / nonfinite metrics | Is training actually progressing and numerically valid? |
| HDBSCAN cluster/alignment diagnostics | What behavioral geometry did the learned embedding create? |
| HDBSCAN centroid-spread metrics | Is the resulting representation useful for the current downstream anomaly-evaluation objective? |

Do not use one category as a substitute for another.

In particular:

```text
better-looking training diagnostics
        ≠
proven downstream improvement
```

A numerically cleaner intervention that materially worsens HDBSCAN evaluation should not be classified as a model improvement solely because its gradients, clipping, loss curve, or activation scale look more conventional.

Conversely, a downstream-better experiment should not be rejected solely because an intermediate diagnostic looks less aesthetically ideal, unless the diagnostic indicates a real correctness or stability problem such as:

- non-finite loss,
- non-finite gradients,
- failed optimizer steps,
- broken execution,
- unreliable evidence.

Training diagnostics should be used to explain mechanisms and generate hypotheses.

HDBSCAN evaluation should be used as the current downstream model-selection evidence.

---

## 19. HDBSCAN Metric Trade-Offs

Do not optimize a single HDBSCAN score blindly.

Model-selection decisions should consider:

- ROC-AUC,
- PR-AUC,
- best F1,
- precision,
- recall / fraud-detection rate,
- benign false-positive rate,
- important fraud-pattern flag rates,
- important benign-pattern false-positive rates.

Examples of meaningful trade-offs include:

```text
higher fraud recall
but
higher benign FP rate
```

or:

```text
better low-and-slow detection
but
worse ATO detection
```

or:

```text
better aggregate PR-AUC
but
large false-positive regression on Traveler
```

If the project has not defined a formal business utility function or mandatory operating-point constraint, report such trade-offs explicitly rather than inventing a universal weighting.

---

## 20. Evaluation Boundary

HDBSCAN evaluation is downstream evidence for the project's current generated evaluation setup.

It does not by itself establish:

- production fraud performance,
- real-world generalization,
- performance on unseen external datasets,
- business impact,
- production operating thresholds.

Use precise claims such as:

```text
improved current HDBSCAN evaluation
```

or:

```text
improved centroid-spread separation on the generated evaluation setup
```

rather than:

```text
improved production fraud detection
```

unless stronger external evidence exists.

---

## 21. Explicitly Out of Scope

The baseline diagnostic profile does not currently provide module-level runtime evidence for:

- parameter-state diagnostics,
- parameter-update diagnostics,
- backward-flow diagnostics,
- disabled child-module diagnostics.

These categories may exist in the probe implementation but are disabled by the baseline diagnostic policy.

The current MCP-controlled experiment boundary does not allow the agent to modify:

- fixed `CFG` fields,
- the diagnostic configuration path,
- model source code,
- trainer source code,
- HDBSCAN/evaluation source code,
- other source code.

If diagnosis requires evidence or changes outside this scope, report the limitation and recommend the smallest useful instrumentation or engineering change rather than assuming or applying it.

HDBSCAN evaluation itself also does not replace external validation/generalization evidence.

---

## 22. Practical Agent Rules

When using this reference:

1. call runtime context tools for the active configuration and model state,
2. use `list_metrics` before assuming a training metric exists,
3. use bounded metric queries for detailed training evidence,
4. distinguish implementation facts from runtime observations and hypotheses,
5. use `evaluate_hdbscan` when downstream evidence is necessary for a model-selection decision,
6. verify HDBSCAN `session_id` / `model_step` freshness before comparison,
7. compare both global and pattern-level downstream behavior,
8. use `get_previous_evaluation` only when the prior experiment was validly evaluated before restart,
9. do not treat cleaner optimization metrics as proof of a better model,
10. keep claims proportional to the evidence actually available.
