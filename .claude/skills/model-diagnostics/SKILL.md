---
name: model-diagnostics
description: Diagnose representation-learning behavior by connecting training evidence, learned geometry, downstream HDBSCAN evaluation, and controlled experiments without assuming project implementation details.
---

# Model Diagnostics

## Purpose

Use evidence to reduce uncertainty enough to support a model-selection or engineering decision.

The diagnostic orchestrator chooses actions and manages the experiment lifecycle. This skill defines how to reason about model-training evidence and downstream representation quality.

The invocation prompt supplies:

- project context;
- the evidence-semantics reference;
- the graph structure and query contract;
- the diagnostic objective and constraints.

Do not assume current model structure, feature construction, loss composition, configuration values, evaluation flow, or metric availability. Obtain those facts from the supplied context, runtime, graph, and evidence reference.

## Evidence Roles

Keep these evidence sources distinct:

- **Runtime state**: active configuration, model/session identity, available metrics, tunables, and executable operations.
- **Graph/source evidence**: implemented data flow, dependencies, configuration bindings, operations, intent, and behavior.
- **Evidence reference**: metric names, collection semantics, evaluation fields, freshness, and history behavior.
- **Diagnostic reasoning**: hypotheses and mechanistic interpretations produced from the evidence.

For active values, runtime state is authoritative. For implementation semantics, prefer extracted graph/source facts over static prose. Treat graph nodes marked `basis=inferred` as interpretations that may require source verification when a decision depends on them.

Report material conflicts instead of silently choosing the most convenient source.

## Evidence Model

Use five distinct claim types:

1. **Observation** — directly recorded evidence.
2. **Derived observation** — a deterministic comparison, ratio, or trend calculated from observations.
3. **Hypothesis** — a plausible explanation that predicts observable behavior.
4. **Mechanism** — an explanation of how implementation, optimization, or representation geometry could produce the observation.
5. **Root cause** — a causal conclusion supported by evidence that distinguishes it from important alternatives.

Do not promote correlation, plausibility, or an unusual value directly into a mechanism or root cause.

Use language such as `consistent with`, `supports`, `may contribute to`, and `cannot distinguish between` when causal evidence remains incomplete.

## Objective Hierarchy

Training diagnostics describe the learning process. They are not automatically the model-selection objective.

Reason through this dependency chain using the actual project context:

```text
implementation correctness
        ↓
training behavior
        ↓
learned representation
        ↓
downstream evaluation
        ↓
requested decision
```

Consequently:

- lower loss does not guarantee a better representation;
- smoother gradients do not guarantee better downstream behavior;
- smaller activations do not guarantee better features;
- less clipping does not guarantee better learning;
- cleaner optimization does not override a downstream regression;
- an unusual diagnostic is not automatically a defect.

When training and downstream evidence disagree, preserve and explain the disagreement.

## Hypothesis Formation

Generate hypotheses from observed evidence and verified project structure rather than a fixed failure-mode checklist.

For each material hypothesis, identify:

- the observations it explains;
- contradictory evidence;
- plausible alternatives;
- the implementation path or learning mechanism involved;
- the evidence that would distinguish the alternatives;
- an allowed intervention, if passive evidence is insufficient;
- the expected training-level and downstream results.

Prefer discriminating evidence over more evidence of the same correlation.

Avoid reflexive conclusions such as:

```text
large scale      → normalize
frequent clipping → increase the threshold
large loss term   → reduce its weight
```

First establish that the observation is harmful to correctness, learning, representation geometry, or the downstream objective.

## Metric Reasoning

Interpret a metric according to what it measures and when it was collected.

- Prefer trends and persistent behavior over isolated samples.
- Respect collection cadence and missing records.
- Align steps and model states before comparing evidence.
- Use relative changes and related metrics when raw magnitude is ambiguous.
- Compare raw L2 norms cautiously across differently sized modules; use normalized statistics when reasoning about per-element scale.
- Do not infer optimizer-update magnitude from parameter gradients unless update evidence exists.
- Do not infer prediction quality or calibration from activation or logit scale alone.
- Do not infer architectural redundancy from gradient asymmetry alone.

Numerical-integrity evidence such as non-finite loss, non-finite gradients, failed execution, or unapplied optimizer steps can establish invalid training. Conventional-looking magnitudes alone cannot establish model quality.

## Loss Reasoning

Distinguish:

- the scalar used for backward propagation;
- component losses;
- monitoring-only losses;
- task weights;
- downstream evaluation metrics.

Use the graph and active runtime context to determine the current loss composition. Do not assume that every loss-like metric contributes to the optimized objective.

A large component loss may indicate scale, difficulty, imbalance, or useful learning pressure. Change its weight only when the intervention tests a supported mechanism and downstream evidence can judge the resulting representation.

## Temporal and Comparison Reasoning

An observation belongs to its recorded step, phase, session, configuration, and model state.

Distinguish two comparison questions:

```text
At the same training budget, which configuration learns more effectively?

After each configuration is sufficiently trained, which representation is better?
```

Matched steps support learning-efficiency comparisons. Sufficiently trained states support final model comparison. Neither should be silently substituted for the other.

An evaluation produced before additional model updates is historical evidence, not evidence for the updated model.

## HDBSCAN and Representation Reasoning

HDBSCAN clusters the geometry supplied to it; it does not repair a poor representation.

Keep density clustering distinct from project-defined post-processing:

```text
HDBSCAN
    → cluster and noise assignments

project post-processing
    → optional centroids, radii, labels, alignment, or transaction scores
```

Use the graph to determine which post-processing the current project actually implements.

Important interpretation boundaries:

- a centroid summarizes a cluster but does not preserve its full density, shape, tails, or substructure;
- HDBSCAN noise is not automatically fraud or anomaly;
- high centroid spread is not automatically fraud;
- low centroid spread is not automatically benign;
- cluster count does not establish downstream detection quality;
- label-free cluster formation does not imply that all later evaluation is label-free.

Model-selection evidence should combine, when available:

- aggregate discrimination metrics;
- operating-point precision, recall, and benign false positives;
- important fraud-pattern detection;
- important benign-pattern false positives;
- cluster, noise, alignment, and geometry diagnostics that help explain the outcome.

Explanatory cluster diagnostics do not replace the downstream objective. If aggregate and pattern-level evidence trade off and no utility rule resolves the trade-off, report it rather than inventing a winner.

## Training and Downstream Evidence Together

Use training evidence to determine whether learning is valid, how an intervention changed the mechanism, and whether comparison timing is meaningful.

Use downstream evidence to determine whether the learned representation improved for the supplied evaluation objective.

Typical interpretations include:

```text
training mechanism changes as predicted
+ downstream improves
→ supports a beneficial intervention
```

```text
training looks cleaner
+ downstream worsens
→ optimization changed, but representation utility regressed
```

```text
little loss change
+ downstream improves
→ useful geometry may have changed without a large objective-level signal
```

Do not force all downstream changes to have an explanation in the currently collected training metrics. State the missing evidence when the mechanism remains unresolved.

## Controlled-Experiment Reasoning

Use controlled intervention when it can distinguish an important hypothesis and the runtime authorizes the change.

A useful experiment specifies:

- the hypothesized mechanism;
- the smallest allowed intervention that tests it;
- the predicted training evidence;
- the predicted downstream evidence;
- the comparison state;
- results that strengthen or weaken the hypothesis.

One-variable changes usually support cleaner attribution. A multi-variable result is evidence about the combined intervention until further isolation is performed.

Consider training progress, convergence, evaluation freshness, protocol consistency, structural differences, and random/data variation before assigning causal confidence.

## Confidence and Evaluation Boundary

Apply confidence to a specific claim:

- **HIGH** — directly observed or repeatedly supported with little ambiguity;
- **MEDIUM** — supported, but meaningful alternatives remain;
- **LOW** — current evidence cannot reliably distinguish the explanation.

A high-confidence observation does not imply a high-confidence mechanism.

Downstream evaluation supports conclusions only within the evaluation population, labels, scoring method, and protocol supplied for the run. It does not by itself establish production performance, external generalization, business impact, or a production operating threshold.

## Evidence-Reference Routing

The invocation prompt supplies the evidence-semantics resource path. Read the relevant sections when the corresponding evidence becomes decision-relevant:

| Diagnostic question | Reference section |
| --- | --- |
| Metric availability and cadence | Availability and collection semantics |
| Loss, optimizer, clipping, or step validity | Training-level metrics |
| Segment-level orientation | Training-segment summaries |
| Module scale or numerical integrity | Activation metrics |
| Module learning signal | Parameter-gradient metrics |
| Downstream result or model selection | Downstream evaluation contract |
| Pattern, alignment, cluster, or dataset explanation | Corresponding evaluation-result subsection |
| Historical comparison | Registry and current/previous semantics |
| Evidence not currently collected | Unavailable evidence |

Do not read every reference section by default. Do not use the reference as a mandatory checklist.

