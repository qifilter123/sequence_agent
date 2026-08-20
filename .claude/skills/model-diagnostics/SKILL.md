---
name: model-diagnostics
description: Evidence-driven methodology for diagnosing PyTorch representation-learning behavior using runtime context, recorded diagnostics, downstream HDBSCAN evaluation, and controlled experiments.
---

# Model Diagnostics

## Goal

Diagnose model-training and representation-learning behavior through evidence rather than a fixed checklist.

A useful diagnosis should reduce uncertainty enough to support a practical engineering decision.

The orchestrator decides which runtime action to take. This skill defines how training diagnostics, downstream HDBSCAN evidence, source semantics, and controlled experiments should be reasoned about.

For this project, model training and CFG2 tuning are not ends in themselves. Their practical purpose is to learn sequence representations that perform well under the current downstream HDBSCAN / centroid-spread evaluation.

Use the following conceptual chain:

```text
implementation
    │
    ▼
training behavior
    │
    ▼
learned representation
    │
    ▼
HDBSCAN cluster geometry
    │
    ▼
transaction centroid spread
    │
    ▼
downstream anomaly-detection quality
```

Training diagnostics explain how the model learns.

HDBSCAN evaluation provides downstream evidence about whether the resulting representation is useful for the current project objective.

Exact metric names, baseline diagnostic categories, collection cadence, HDBSCAN result fields, and metric semantics are documented in:

`references/metric-categories.md`

Use that reference as a capability and semantics map, not as a mandatory sequence of checks.

Runtime context is authoritative for the active experiment.

---

## Evidence Model

Maintain a strict distinction between:

1. **Observation**
   - A value, state, or trend directly present in recorded runtime or evaluation evidence.

2. **Derived observation**
   - A deterministic comparison, ratio, trend, or relationship calculated from observations.

3. **Hypothesis**
   - A plausible explanation that predicts observable behavior.

4. **Mechanistic interpretation**
   - A proposed explanation of how model structure, optimization, representation geometry, or downstream scoring produces the observations.

5. **Root cause**
   - A causal conclusion supported by evidence that meaningfully distinguishes it from important alternatives.

Do not promote correlation directly into mechanism or root cause.

A high-confidence observation does not imply a high-confidence causal explanation.

When causal evidence is insufficient, prefer language such as:

- "consistent with"
- "supports the hypothesis that"
- "may contribute to"
- "cannot distinguish between"

Examples:

```text
Observation:
time_emb input abs_max is very large.

Hypothesis:
large time values may create difficult optimization dynamics.

Not yet justified:
large time values are harmful and should be normalized.
```

Likewise:

```text
Observation:
Traveler flag rate increased after an experiment.

Hypothesis:
the representation may have lost invariance to legitimate travel behavior.

Not yet justified:
HDBSCAN itself is defective.
```

---

## Objective Hierarchy

Distinguish intermediate optimization health from downstream model usefulness.

For the current project:

```text
training objective
    = representation-learning mechanism

HDBSCAN evaluation
    = current downstream model-selection evidence
```

Therefore:

- lower training loss does not automatically imply a better model,
- smoother gradients do not automatically imply a better model,
- smaller activations do not automatically imply a better model,
- less clipping does not automatically imply a better model,
- a more conventional-looking optimization trace does not automatically imply a better embedding.

A training intervention should be judged at two levels:

1. **Did it change training behavior in the intended way?**
2. **Did the resulting representation improve downstream HDBSCAN behavior?**

When those levels disagree, preserve the distinction.

Examples:

```text
training improves
HDBSCAN improves
    → evidence supports a beneficial intervention
```

```text
training improves
HDBSCAN worsens
    → optimization became cleaner, but representation quality worsened
```

```text
training changes little
HDBSCAN improves materially
    → representation geometry may have improved without a large loss-level signal
```

```text
training looks less ideal
HDBSCAN improves
    → do not reject the model solely for diagnostic aesthetics
      unless training is actually invalid or unstable
```

The end goal is not to maximize a single HDBSCAN number blindly.

Use aggregate and pattern-level downstream evidence together.

---

## Expert ML Reasoning

Apply general machine-learning knowledge when forming hypotheses and designing experiments.

Do not wait for this skill, project documentation, or the runtime tunable list to enumerate every relevant ML mechanism, hyperparameter, or experimental methodology.

Reason broadly about factors that may affect training or representation quality, including areas such as:

- optimization,
- sampling and batching,
- convergence and training duration,
- learning-rate behavior,
- regularization,
- model capacity,
- loss balance,
- feature and representation construction,
- temporal modeling,
- interactions between these mechanisms.

This list is illustrative, not exhaustive.

Project documentation defines project-specific semantics, implementation facts, and constraints.

It does not replace general ML reasoning.

The runtime tunable allowlist determines which interventions may actually be executed.

It should not restrict which hypotheses or candidate interventions may be considered.

Use the following reasoning pattern:

```text
observed evidence
      │
      ▼
ML mechanism hypothesis
      │
      ▼
candidate intervention
      │
      ▼
runtime authority check
      │
      ▼
controlled experiment
      │
      ▼
training evidence
      +
downstream HDBSCAN evidence
```

Generate candidate interventions from observed evidence and plausible mechanisms rather than from a fixed hyperparameter checklist.

Do not perform broad hyperparameter search merely because tunables are available.

Prefer interventions that:

- test an important hypothesis,
- reduce meaningful uncertainty,
- or provide a plausible path to improving downstream representation quality.

The purpose of ML expertise is not to try every known technique.

It is to choose the most informative or promising next experiment from the current evidence.

---

## Hypothesis Formation

Generate hypotheses from the actual runtime context and observed evidence.

Do not begin with a mandatory catalog of failure modes.

For an important hypothesis, consider:

- what supports it,
- what contradicts it,
- what alternatives remain plausible,
- what observable result would distinguish those alternatives,
- whether additional observation or a controlled intervention would provide better evidence,
- whether downstream HDBSCAN evaluation is needed to decide whether the intervention is actually useful,
- whether collecting that evidence is worth the cost.

Treat explanations that cannot currently be tested as unresolved.

Prefer discriminating evidence over additional evidence that merely confirms the same correlation.

Avoid reflexive transformations such as:

```text
large scale
    → normalize

high clipping
    → increase clip threshold

large loss component
    → reduce its weight
```

until evidence shows that the observed behavior is harmful to the relevant training or downstream objective.

---

## Evidence Selection

Prefer evidence with high information value.

Useful forms of evidence include:

- trends rather than isolated points,
- persistent behavior rather than one-off observations,
- related-module comparisons,
- earlier-versus-later training comparisons,
- relative changes rather than raw magnitude alone,
- agreement or disagreement across metric categories,
- matched current/previous experiment comparisons,
- controlled before/after interventions,
- HDBSCAN global metric changes,
- HDBSCAN per-pattern changes,
- changes in cluster geometry that explain downstream behavior.

Do not collect evidence without knowing what uncertainty it is intended to reduce.

Do not run HDBSCAN after every small training segment merely because the tool is available.

Run downstream evaluation when its result can materially affect:

- model selection,
- acceptance/rejection of a tuning hypothesis,
- interpretation of a representation-changing intervention,
- comparison with a baseline or previous experiment.

---

## Metric Interpretation

Interpret metrics according to what they actually measure.

Do not impose universal thresholds for terms such as:

- small,
- large,
- collapsed,
- exploding,
- inactive,
- unstable.

Interpret values relative to:

- training history,
- module role,
- neighboring modules,
- parameter count,
- related metrics,
- sampling cadence,
- experiment configuration,
- downstream evaluation behavior.

When comparing differently sized modules, raw L2 norms may not be directly comparable.

Prefer normalized statistics such as RMS or mean absolute magnitude when the question concerns per-element scale.

A metric can identify an unusual condition without proving that the condition is harmful.

---

## Interpretation Boundaries

Do not infer properties that a metric does not measure.

Examples:

- logit magnitude is not calibration,
- regression output scale is not prediction quality,
- parameter-gradient magnitude is not optimizer-update magnitude,
- gradient clipping frequency is not proof that clipping harms convergence,
- module gradient asymmetry is not proof of architectural redundancy,
- activation sparsity does not by itself identify its mechanism,
- lower training loss does not establish better downstream representation quality,
- a cluster count does not establish anomaly-detection quality,
- a high centroid spread does not automatically mean fraud,
- a low centroid spread does not automatically mean benign behavior.

Use such observations to form hypotheses, not to skip causal reasoning.

---

## Loss Semantics

Do not assume every loss-like metric participates in the optimized objective.

Distinguish between:

- the scalar used for `backward()`,
- task-level loss components,
- raw monitoring metrics,
- downstream evaluation metrics.

If objective composition is not explicitly available from runtime context, source, or metric documentation, treat it as unknown.

A large task-loss component does not automatically mean its task weight should be reduced.

That task may be supplying useful representation-learning pressure.

Before changing objective balance, consider whether the task contributes useful downstream structure.

---

## Temporal Reasoning

Respect metric sampling cadence.

When combining evidence:

- align records to the same or comparable training steps when possible,
- distinguish point samples from segment-wide behavior,
- do not treat sparse samples as dense evidence,
- do not claim persistent or monotonic behavior from too few observations.

Confidence should reflect sampling density.

The same principle applies to experiment comparison: compare equivalent or matched training progress whenever possible.

HDBSCAN evaluations also belong to an exact model state.

An evaluation from model step `N` must not be treated as evaluation of the model after further optimizer steps.

---

## Adaptive Training Horizon

Treat training duration as an experimental decision rather than a fixed constant.

Do not assume that a previously used number of steps is automatically sufficient for every model configuration.

Use observed learning dynamics and downstream evaluation to determine whether additional training is informative.

Consider extending training when evidence suggests that:

- important loss components are still evolving,
- representation-related diagnostics are still changing materially,
- downstream HDBSCAN quality may still be improving,
- the model has not reached a meaningful comparison state,
- an experimental configuration appears to converge at a different rate from the baseline,
- an apparent regression may simply reflect slower convergence.

Do not assume that identical short training windows always provide a fair model comparison when one experiment is clearly still learning.

Matched training steps are useful for causal comparison, but sufficiently converged model states may also be necessary for model selection.

Distinguish these two questions:

```text
At the same training budget,
which configuration learns more effectively?

versus

After each configuration has trained sufficiently,
which representation is better?
```

Both can be useful, but they answer different questions.

Likewise, do not continue training merely because more steps are available.

Stop extending the training horizon when:

- relevant training signals have stabilized,
- downstream HDBSCAN results are no longer changing materially,
- additional optimization is unlikely to alter the model-selection conclusion,
- or the expected information value of more training is low.

Training duration is therefore part of experimental design even when it is not a CFG2 tunable.

---

# HDBSCAN and Downstream Representation Reasoning

## HDBSCAN Overview

HDBSCAN is a hierarchical density-based clustering method.

At a high level, it attempts to identify regions of embedding space that contain sufficiently persistent dense structure while allowing some points to remain unassigned as noise.

Unlike methods such as k-means, HDBSCAN does not require the number of clusters to be specified in advance.

Conceptually:

```text
record embeddings
      │
      ▼
local density relationships
      │
      ▼
hierarchical cluster structure
      │
      ▼
stable density-based clusters
      │
      ├── cluster 0
      ├── cluster 1
      ├── cluster 2
      └── noise
```

HDBSCAN operates on the geometry supplied by the model.

It does not repair a poor embedding.

If behaviorally different records collapse into the same region, clustering may not separate them.

If legitimate variants are scattered into distant regions, clustering may expose that separation and downstream scoring may turn it into false positives.

Therefore HDBSCAN results are evidence about both:

- the clustering procedure,
- the geometry learned by the sequence encoder.

---

## HDBSCAN Does Not Directly Produce Centroids

Keep the distinction precise:

```text
HDBSCAN
    → cluster assignments / noise assignments

project post-processing
    → centroid for each retained cluster
```

For a cluster containing embeddings:

```text
C_k = {x_1, x_2, ..., x_n}
```

the project can summarize that cluster with a centroid:

```text
μ_k = mean(x_i for x_i in C_k)
```

The centroid is therefore a project-defined summary of a learned HDBSCAN cluster.

It is not the object that HDBSCAN itself optimizes.

This matters because a density-based cluster can have a non-spherical or irregular shape.

A centroid compresses that shape into one representative point.

Use centroid-based diagnostics as a useful approximation of behavioral regions, not as proof that the entire cluster is geometrically spherical or homogeneous.

---

## Reference Centroid Formation

The current project uses a reference-style phase to build behavioral centroids.

Conceptually:

```text
reference / historical-style transactions
        │
        ▼
multiple anchored sequence views
        │
        ▼
trained encoder
        │
        ▼
record embeddings
        │
        ▼
HDBSCAN fit
        │
        ▼
cluster assignments
        │
        ▼
centroid per retained cluster
```

The HDBSCAN fit uses embedding geometry.

Generated fraud/scenario labels are not required to create the density-based cluster assignments themselves.

However, the current evaluator may use generated labels after clustering for diagnostic interpretation such as:

- cluster fraud proportion,
- FRAUD/NORMAL assignment,
- semantic summaries,
- selection of fraud centroids for alignment analysis.

Therefore distinguish:

```text
unsupervised cluster formation
```

from:

```text
label-informed post-cluster evaluation / interpretation
```

Do not describe the entire evaluation pipeline as fully label-free when post-cluster labels are used.

---

## Why Centroids Can Help Detect Anomaly

A transaction may generate multiple behavioral views, for example from different anchors or relational perspectives.

If those views are behaviorally consistent, their embeddings may map to the same or nearby behavioral regions.

Conceptually:

```text
benign coherent transaction

view A ──► cluster 3
view B ──► cluster 3
view C ──► cluster 4

cluster 3 and cluster 4 are nearby
        ↓
small centroid spread
```

An anomalous transaction may produce views that map into substantially different learned behavioral regions:

```text
behaviorally inconsistent transaction

view A ──► cluster 2
view B ──► cluster 7
view C ──► cluster 11

centroids are far apart
        ↓
large centroid spread
```

The current project uses intra-transaction centroid spread as a downstream anomaly signal.

A generic form is:

```text
spread(transaction)
    =
max distance(μ_i, μ_j)
over distinct centroids reached by its views
```

Large spread can indicate that different views of the same transaction disagree strongly about its behavioral context.

This can help expose:

- account-takeover transitions,
- credential-stuffing behavior,
- abrupt entity changes,
- mixed behavioral regimes,
- other anomalies that create inconsistent sequence views.

But centroid spread is not intrinsically a fraud score in every domain.

Legitimate behavior can also be multi-modal.

Therefore false-positive behavior on legitimate patterns is an important part of evaluation.

---

## HDBSCAN Global Evaluation

Use aggregate downstream metrics to understand overall anomaly separation.

Typical evidence includes:

- ROC-AUC,
- PR-AUC,
- best F1,
- precision,
- recall,
- benign false-positive rate.

Interpret them jointly.

Examples:

```text
higher recall
+
much higher benign FP
    → trade-off, not automatic improvement
```

```text
higher PR-AUC
+
stable benign FP
+
no important pattern regression
    → stronger evidence of improvement
```

A threshold chosen to maximize F1 on the current evaluation labels is an evaluation operating point.

It is not automatically a production threshold.

---

## HDBSCAN Pattern-Level Evaluation

Aggregate metrics can hide important representation failures.

Inspect fraud and benign patterns separately when available.

Pattern-level evidence can reveal cases such as:

```text
overall score slightly improves
but
one important fraud pattern collapses
```

or:

```text
overall score remains similar
but
Traveler false positives increase dramatically
```

or:

```text
low-and-slow detection improves
while
ATO recall decreases
```

These are representation trade-offs that should influence model selection.

If no business utility function specifies how such trade-offs should be weighted, report them explicitly rather than inventing a universal ranking.

---

## Cluster and Alignment Diagnostics

Cluster-level evidence can help explain why downstream scores changed.

Useful questions include:

- Did the number of stable clusters change materially?
- Did noise increase substantially?
- Did formerly coherent benign behavior fragment across clusters?
- Did fraud patterns become more concentrated or more dispersed?
- Are different transaction views mapping to more distant centroids?
- Did anchor-specific clustering become dominant?
- Did alignment to fraud-associated centroids change?

These are explanatory diagnostics.

They should support downstream interpretation, not replace global and pattern-level anomaly metrics.

---

## Important HDBSCAN Caveats

### Representation Dependence

HDBSCAN can only cluster the geometry it receives.

A downstream regression after a model change may result from altered embedding geometry even when training diagnostics look healthier.

### Hyperparameter Dependence

Results can depend on evaluation settings such as:

- `min_cluster_size`,
- `min_samples`,
- distance metric,
- cluster-selection method,
- evaluation dataset generation.

When comparing model experiments, keep the evaluation protocol fixed unless the evaluation method itself is the controlled variable.

### Density and Centroid Are Different Concepts

HDBSCAN is density-based.

Centroid spread is centroid-based.

A centroid is a compressed summary of a density cluster and can lose information about:

- shape,
- density variation,
- substructure,
- tails.

Do not infer that two records are behaviorally equivalent solely because they are nearest to the same centroid.

### Noise Is Not Automatically Fraud

HDBSCAN noise means the point was not assigned to a sufficiently stable density cluster under the current settings.

Noise is not inherently anomalous or fraudulent.

### High Spread Is Not Automatically Fraud

Legitimate multi-modal behavior can produce large spread.

Always examine benign false-positive patterns.

### Low Spread Is Not Automatically Benign

A coordinated anomaly can remain internally consistent and map repeatedly to one anomalous region.

Low spread alone cannot prove benign behavior.

---

## Diagnostic Abnormality vs Useful Signal

An unusual activation scale, gradient magnitude, clipping rate, or other optimization pattern is not automatically a defect.

Before recommending removal, normalization, clipping changes, or feature transformation, ask:

1. Is there evidence that the behavior impairs training correctness or stability?
2. Is there evidence that it impairs the learned representation?
3. Could the unusual signal contain useful behavioral information?
4. What controlled experiment can distinguish these possibilities?
5. What HDBSCAN result would show that the intervention helped or harmed the end goal?

Treat diagnostic abnormalities as investigation targets, not automatic optimization targets.

A representation-changing intervention should normally be judged with downstream evaluation.

---

## Controlled Experiment Reasoning

Use a controlled intervention when passive evidence cannot sufficiently distinguish important hypotheses and the runtime exposes an appropriate tunable parameter.

Candidate interventions should come from ML reasoning first.

The runtime allowlist is then used to determine which candidates are executable.

The runtime-declared tunable allowlist is authoritative for actions.

Do not infer that a parameter is experimentally mutable merely because it exists in configuration or source code.

Before an experiment, determine:

- the hypothesis being tested,
- the tunable intervention,
- the expected training observation,
- the expected downstream HDBSCAN observation,
- the comparison point,
- what result would strengthen the hypothesis,
- what result would weaken it.

Prefer the minimum number of changed tunables needed to answer the question.

A one-variable intervention generally supports cleaner attribution.

When several tunables change together, treat the result as evidence about the combined intervention unless further experiments separate their effects.

Controlled experiments strengthen causal reasoning but do not eliminate all alternative explanations.

Consider:

- training-window alignment,
- convergence state,
- sampling density,
- model-structure differences,
- random/data variation where relevant,
- contradictory metric behavior,
- whether the observed change is persistent.

---

## Factor Isolation

When a proposed change affects multiple mechanisms, decompose it into smaller interventions when practical.

For two conceptually separate changes A and B:

```text
baseline       A0 B0
A only         A1 B0
B only         A0 B1
combined       A1 B1
```

This supports reasoning about:

- the effect of A,
- the effect of B,
- interaction between A and B.

Do not attribute a multi-change result to one component without discriminating evidence.

This principle applies to:

- feature transformations,
- model-path changes,
- loss changes,
- optimizer changes,
- downstream evaluation changes.

---

## Current / Previous Comparisons

A valid comparison requires evidence from two actual experiments.

Do not invent a baseline comparison when no previous experiment exists.

For current-versus-previous evidence:

- identify the exact configuration delta,
- use the same metric definitions,
- prefer matched training windows when answering learning-efficiency questions,
- consider sufficiently converged states when answering final model-quality questions,
- verify HDBSCAN evaluations correspond to the intended model steps,
- keep the HDBSCAN protocol fixed unless it is itself under test,
- compare related signals rather than one isolated metric,
- compare global and pattern-level downstream evidence,
- separate observation from causal interpretation.

A compact comparison summary may identify useful differences, but detailed claims should be supported by underlying metrics when necessary.

Remember that a runtime may retain only the current and immediately previous experiment.

Across a longer experiment sequence, model selection should still consider the best previously observed downstream result rather than only the latest pairwise comparison.

---

## Training and HDBSCAN Together

Use training evidence and downstream evidence jointly.

Training diagnostics can answer questions such as:

- Is optimization progressing?
- Which task dominates the loss?
- Where are gradients concentrated?
- Is clipping persistent?
- Are activations numerically valid?
- Did a tunable change the intended training mechanism?
- Has the model likely trained long enough for the comparison being made?

HDBSCAN can answer different questions:

- Did embedding geometry improve for the downstream anomaly task?
- Did fraud/benign separation improve?
- Did important fraud-pattern detection improve?
- Did benign false positives worsen?
- Did representation fragmentation or cluster geometry change?

A strong diagnosis connects the two layers without confusing them.

Example:

```text
Observation:
a tunable reduces gradient clipping.

Observation:
HDBSCAN PR-AUC decreases and benign FP rises.

Conclusion:
the tunable improved one optimization statistic but did not improve
the current downstream representation objective.
```

Another example:

```text
Observation:
experiment B has worse HDBSCAN at an early matched step.

Observation:
its training losses are still improving materially while baseline A had
already stabilized at that step.

Conclusion:
the matched-step comparison establishes slower learning at that budget,
but does not yet establish that B's sufficiently trained representation
is worse.
```

---

## Evaluation Boundary

HDBSCAN evaluation is independent downstream evidence relative to training diagnostics, but it still belongs to the project's current generated evaluation setup.

It can support claims about:

- current embedding geometry,
- current HDBSCAN cluster behavior,
- current centroid-spread anomaly separation,
- comparison among controlled model experiments under the same evaluation protocol.

It does not by itself establish:

- production fraud performance,
- real-world generalization,
- unseen external-data performance,
- business impact,
- production operating thresholds.

Prefer precise claims such as:

- "improved current HDBSCAN evaluation",
- "improved centroid-spread separation",
- "reduced benign false positives in the generated evaluation".

Do not automatically translate those claims into production guarantees.

---

## Missing Evidence

If current evidence and allowed experiments cannot distinguish important explanations, identify:

- **Unresolved question**
- **Missing evidence**
- **Why current evidence is insufficient**
- **Minimal additional instrumentation or engineering change**
- **What result would distinguish the competing hypotheses**

Do not invent unavailable evidence.

Changes outside the runtime's declared tunable boundary should remain recommendations rather than actions.

If HDBSCAN reveals a downstream regression but available training diagnostics cannot explain the mechanism, report that distinction rather than forcing a training-based explanation.

---

## Confidence

Apply confidence to the specific claim being made.

Use:

- **HIGH** — directly observed or persistent evidence with little ambiguity,
- **MEDIUM** — interpretation is supported but meaningful alternatives remain,
- **LOW** — explanation cannot be reliably distinguished with current evidence.

Do not give a high-confidence causal conclusion merely because the underlying observation is high confidence.

Controlled interventions can increase causal confidence when they clearly discriminate between alternatives, but confidence must still reflect:

- experiment design,
- matched or appropriate training progress,
- convergence state,
- evaluation freshness,
- downstream protocol consistency,
- remaining confounders.

---

## Stopping

Stop collecting evidence when:

- the evidence supports a practical conclusion,
- the best supported current model/configuration is clear under the downstream objective,
- the next engineering action is clear,
- a controlled experiment has sufficiently narrowed the important hypotheses,
- additional training, HDBSCAN evaluation, or allowed experiments are unlikely to materially change the conclusion,
- or current instrumentation and experimental boundaries cannot resolve the remaining uncertainty.

Do not stop merely because training diagnostics look healthy when downstream model quality remains unresolved.

Do not continue experimenting solely because more tunables or training budget remain.

Recommendations must remain proportional to the evidence.