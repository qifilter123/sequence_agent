---
document_role: evidence-semantics
schema_version: 1
runtime_contract: diagnostics-v1
---

# Model Diagnostic Evidence Contract

## Purpose and Authority

This reference defines the diagnostic metrics and evaluation fields exposed by the current diagnostics runtime. It is a field-semantics and availability contract, not a diagnostic workflow and not a description of the current model implementation.

Use:

- runtime context for active configuration, structure, session, and model state;
- the project graph for feature, model, loss, and evaluation implementation;
- this reference for metric/evaluation field meaning, collection, freshness, and history behavior;
- the diagnostic methodology for interpretation and hypothesis formation.

A documented metric may be absent from a particular session. Discover runtime availability before querying it.

## Availability and Collection Semantics

The current baseline diagnostic profile collects:

- training-level metrics on each attempted iteration when execution reaches their recording point;
- activation input/output metrics for enabled logical modules every 50 diagnostic steps;
- parameter-gradient metrics for enabled logical modules every 50 diagnostic steps.

Step `0` is a scheduled module-level collection step. Restarting creates a new metric registry and probe state.

The baseline profile does not collect module-level:

- parameter-state metrics;
- parameter-update metrics;
- backward-flow metrics;
- metrics for disabled child modules.

Do not assume that a module is instrumented merely because it exists in the model structure. Use runtime metric discovery to determine actual availability.

A documented metric may not exist because:

- its first scheduled collection point has not occurred;
- the logical module is not enabled;
- the active trainer uses a compatible contract that does not emit it;
- execution failed before its recording point.

## Training-Level Metrics

Training-level metrics use stage `training`.

### Loss

| Metric | Semantics |
| --- | --- |
| `loss.total` | Scalar used for backward propagation. Obtain its current composition from runtime context and the graph. |
| `loss.recon` | Reconstruction objective exposed by the active trainer. Its current component composition is defined by the graph. |
| `loss.v` | Normalized time-related prediction loss before its task-specific weight. Conditional on the active trainer return contract. |
| `loss.sw` | Normalized switch-pattern loss before its task-specific weight. Conditional on the active trainer return contract. |
| `loss.amt` | Normalized amount-related prediction loss before its task-specific weight. Conditional on the active trainer return contract. |
| `loss.is_new` | Normalized multi-label is-new loss before its task-specific weight. Conditional on the active trainer return contract. |
| `loss.sw_ce` | Raw switch-class cross-entropy averaged over valid tokens. Available under both supported trainer return contracts. |
| `loss.nonfinite` | Boolean: `not isfinite(loss.total)`. Numerical-integrity signal, not a quality metric. |

Do not assume `loss.sw` and `loss.sw_ce` are semantically identical. Verify the active graph when the distinction matters.

### Optimizer and Global Gradient

| Metric | Semantics |
| --- | --- |
| `optimizer.lr` | Learning rate from the first optimizer parameter group. |
| `grad.global.l2_pre_clip` | Global gradient norm returned by `clip_grad_norm_`; represents the norm before clipping. |
| `grad.global.nonfinite` | Boolean indicating whether the measured pre-clipping global norm is non-finite. |
| `grad.clip.max_norm` | Configured clipping threshold, not a measured post-clipping norm. |
| `grad.clip.triggered` | Boolean: finite pre-clipping norm exceeded `grad.clip.max_norm`. |
| `optimizer.step_applied` | Whether an optimizer update was applied for the attempted step. |

Module parameter-gradient probes are collected after backward propagation and before clipping.

A failed or partial step must not be treated as a parameter update unless `optimizer.step_applied` is true.

## Training-Segment Summaries

`train_segment` returns compact tool-result summaries rather than ordinary metric-history records. A result may include:

- status and structured error information;
- segment identity;
- start/end steps;
- requested/completed steps;
- elapsed time;
- summary statistics for available losses;
- pre-clipping global gradient statistics;
- clipping frequency and threshold.

Use detailed metric queries when temporal shape or individual records matter.

## Activation Metrics

Enabled logical modules can record the following for both `activation.input` and `activation.output`:

| Suffix | Semantics |
| --- | --- |
| `.shape` | Runtime tensor shape captured by the forward hook. |
| `.mean` | Mean of finite values. |
| `.std` | Standard deviation of finite values. |
| `.rms` | Root-mean-square magnitude of finite values. |
| `.abs_max` | Maximum absolute finite value. |
| `.zero_frac` | Fraction of all elements that are finite and exactly zero. |
| `.nonfinite_frac` | Fraction of all elements that are NaN or Inf. |

Non-finite elements are excluded from finite-value reductions.

These metrics measure tensor shape, scale, dispersion, sparsity, and numerical integrity. They do not directly measure prediction quality, calibration, or downstream representation utility.

## Parameter-Gradient Metrics

Enabled logical modules can record:

| Metric | Semantics |
| --- | --- |
| `grad.param.l2` | Total L2 norm across collected trainable-parameter gradients. |
| `grad.param.rms` | RMS magnitude across collected gradient elements. |
| `grad.param.abs_mean` | Mean absolute gradient magnitude. |
| `grad.param.abs_max` | Maximum absolute finite gradient value. |
| `grad.param.coverage` | Gradient parameter elements divided by trainable parameter elements. |
| `grad.param.nonfinite_frac` | Fraction of gradient elements that are NaN or Inf. |

These metrics describe gradient signal before clipping. They do not measure optimizer-update magnitude.

Raw L2 magnitude depends on module size. RMS and absolute mean are more suitable for per-element scale comparisons across differently sized modules.

## Downstream Evaluation Contract

Use `evaluate_hdbscan` for the active in-memory model. Evaluation results are maintained separately from the training metric registry.

The graph defines how the evaluation dataset, embeddings, clusters, centroids, alignment, scores, and reports are produced. This reference defines only the returned evidence contract.

### Identity, Status, and Freshness

A result can include:

- `status`;
- `session_id`;
- `session_step`;
- `model_step`;
- `evaluated_at`;
- `elapsed_seconds`;
- effective model configuration under `cfg2`;
- effective evaluator configuration under `config`.

A successful optimizer update changes model state and invalidates the previous current-model evaluation. Use an evaluation for model selection only when its session and model step match the intended comparison state.

A failed result includes structured error information and is not valid model-quality evidence.

### Global Centroid-Spread Result

The main transaction-level result is returned under `centroid_spread`.

| Field | Semantics |
| --- | --- |
| `roc_auc` | ROC-AUC of the transaction score against generated labels. |
| `pr_auc` | Precision-recall AUC with fraud/anomaly as the positive class. |
| `best_f1` | Best F1 found over evaluated score thresholds. |
| `threshold` | Threshold associated with `best_f1`; evaluation operating point, not a production threshold. |
| `precision` | Precision at the best-F1 threshold. |
| `recall` | Recall at the best-F1 threshold. |
| `accuracy` | Accuracy at the best-F1 threshold. |
| `fraud_detection_rate` | `TP / (TP + FN)`; currently equivalent to positive-class recall. |
| `benign_fp_rate` | `FP / (FP + TN)`. |
| `confusion` | `tp`, `fp`, `fn`, and `tn` at the best-F1 threshold. |

Population fields include:

- `transactions_total`;
- `fraud_transactions`;
- `benign_transactions`.

Threshold magnitude alone is not a model-quality measure because score geometry can change between representations.

### Pattern-Level Result

Pattern results are returned under `per_pattern`. Each item can include:

| Field | Semantics |
| --- | --- |
| `scenario` | Generated scenario name. |
| `n` | Evaluated transaction count for the scenario. |
| `is_fraud` | Whether the generator classifies the scenario as fraud/anomaly. |
| `role` | `detect` for fraud/anomaly patterns or `FP-rate` for benign patterns. |
| `mean_centroid_spread` | Mean transaction score for the scenario; not a thresholded detection rate. |
| `flag_rate` | Fraction at or above the global best-F1 threshold. Detection rate for fraud patterns and false-positive rate for benign patterns. |
| `suppressed_rate` | Fraction suppressed by evaluator logic when such logic is active. Presence of the field does not prove suppression is active. |

### Alignment Result

Alignment diagnostics are returned under `alignment` and can include:

- `available` and an unavailable reason;
- `pure_cluster_count`;
- `fraud_centroid_count`;
- `transactions_total`;
- `transactions_aligned`;
- `fraud_aligned`;
- `benign_aligned`;
- `per_scenario` entries containing scenario identity, counts, role, and `alignment_rate`.

Alignment is explanatory geometry evidence. It is not the primary transaction-score result unless the invocation objective explicitly makes it so.

### Cluster-Build Result

Cluster diagnostics are returned under `cluster_build` and can include:

- `cluster_count`;
- `fraud_cluster_count`;
- `normal_cluster_count`;
- `noise.count`;
- `noise.n_anomaly`;
- `noise.top_scenarios`;
- per-cluster summaries.

Per-cluster summaries can include:

- `label`, `size`, `n_anomaly`, and `n_normal`;
- `p_fraud`, `purity`, `assigned`, and `semantic_type`;
- `anchor_purity`, `anchor_dominant`, and `anchor_mix`;
- `dominant_scenarios`;
- `max_align_distance` and `median_align_distance`.

Centroid vectors are omitted from MCP transport.

### Evaluation-Dataset Result

Population information is returned under `evaluation_dataset` and can include:

- `transactions`;
- `record_count_with_len_gate`;
- `record_count_all`;
- `fraud_transactions`;
- `benign_transactions`.

Use these fields and effective `config` to determine whether evaluation populations and protocols are comparable.

## Registry and Query Semantics

The `DiagnosticRegistry` stores training and module metrics by:

- `stage`;
- `metric_name`.

Each record carries:

- `timestamp`;
- `step`;
- `epoch`;
- `phase`;
- `tags`.

Use `list_metrics` before assuming a metric exists. Use bounded `query_metrics` calls for detailed histories.

Evaluation results are not queried through the training metric registry.

## Current and Previous Evidence

A successful controlled restart can retain evidence from the immediately replaced experiment:

- previous configuration;
- previous training metrics;
- previous model structure;
- previous evaluation, if a valid final-state evaluation existed before replacement.

Previous training evidence does not imply that a previous evaluation exists.

`get_experiment_comparison` can expose compact current/previous evidence. Global evaluation deltas can include:

- `roc_auc`;
- `pr_auc`;
- `best_f1`;
- `precision`;
- `recall`;
- `accuracy`;
- `fraud_detection_rate`;
- `benign_fp_rate`.

Pattern deltas can include:

- `mean_centroid_spread`;
- `flag_rate`;
- previous/current sample counts and scenario identity.

Threshold and confusion values remain available in full evaluation results but are not part of the compact delta block.

The runtime retains only current and immediately previous experiment evidence. Longer-horizon best-so-far tracking belongs to the orchestrator.

## Unavailable Evidence

Unless runtime discovery shows otherwise, do not claim module-level evidence for:

- parameter state;
- optimizer update magnitude;
- backward-flow timing;
- disabled child modules.

If a required field or category is not recorded, report it as unavailable rather than inferring its value from adjacent metrics.

This evaluation contract applies only to the current generated evaluation setup. It does not establish production performance, external generalization, business impact, or production thresholds.

