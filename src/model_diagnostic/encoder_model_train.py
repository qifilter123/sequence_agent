from contextlib import nullcontext
from typing import Dict, List, Optional
from pathlib import Path
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

from model_diagnostic import generic_feature_util as feature_util
from model_diagnostic import generic_seq_generator as seq_gen
from model_diagnostic import batch_util
from model_diagnostic.cfg_base import CFG2

from model_diagnostic.diagnostic_probe_manager import DiagnosticProbeManager, StageProbeConfig
from model_diagnostic.diagnostic_registry import DIAGNOSTICS


_FEATURE_DAG_RUNTIME = None
_MODEL_DAG_RUNTIME = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _get_feature_dag_runtime():
    """Lazily load feature DAG configs once per experiment/runtime.

    Training calls ``build_full_features`` for every batch, so YAML parsing must
    stay outside the hot path. Call ``reload_feature_dag_runtime`` when an agent
    or experiment changes DAG YAML in the same Python process.
    """
    global _FEATURE_DAG_RUNTIME
    if _FEATURE_DAG_RUNTIME is None:
        from model_diagnostic.dag.dag_processor import DagProcessor
        from model_diagnostic.dag_feature_ops import FEATURE_DAG_REGISTRY

        processor = DagProcessor(FEATURE_DAG_REGISTRY)
        config_dir = Path(__file__).resolve().parent / "config"
        extractor_cfg = processor.load_config(config_dir / "input_extractor.yaml")
        transformer_cfg = processor.load_config(config_dir / "input_transformer.yaml")
        _FEATURE_DAG_RUNTIME = (processor, extractor_cfg, transformer_cfg)

    return _FEATURE_DAG_RUNTIME


def reload_feature_dag_runtime():
    """Reload feature DAG YAML for a new experiment in the same process."""
    global _FEATURE_DAG_RUNTIME
    _FEATURE_DAG_RUNTIME = None
    return _get_feature_dag_runtime()


def _get_model_dag_runtime():
    """Lazily load the model-builder DAG config once per experiment/runtime.

    Importing ``dag_model_ops`` populates MODEL_BUILDER_REGISTRY, mirroring
    the feature DAG flow where importing ``dag_feature_ops`` populates
    FEATURE_DAG_REGISTRY.
    """
    global _MODEL_DAG_RUNTIME
    if _MODEL_DAG_RUNTIME is None:
        from model_diagnostic.dag.dag_processor import DagProcessor
        from model_diagnostic.dag_model_ops import MODEL_BUILDER_REGISTRY

        processor = DagProcessor(MODEL_BUILDER_REGISTRY)
        config_dir = Path(__file__).resolve().parent / "config"
        model_cfg = processor.load_config(config_dir / "model_structure.yaml")
        _MODEL_DAG_RUNTIME = (processor, model_cfg)

    return _MODEL_DAG_RUNTIME


def reload_model_dag_runtime():
    """Reload model_structure.yaml for a new experiment in the same process."""
    global _MODEL_DAG_RUNTIME
    _MODEL_DAG_RUNTIME = None
    return _get_model_dag_runtime()


def build_full_features(batch, current_cfg, is_training=False):
    processor, extractor_cfg, transformer_cfg = _get_feature_dag_runtime()
    runtime = {
        "cfg": current_cfg,
        "is_training": is_training,
    }

    extracted = processor.run(
        extractor_cfg,
        inputs=batch,
        runtime=runtime,
    )["raw"]

    transformed = processor.run(
        transformer_cfg,
        inputs=extracted,
        runtime=runtime,
    )

    return transformed["model_input"], transformed["raw"]


def compute_nextstep_recon_loss(preds, raw, current_cfg):

    mask_src = raw["mask"][:, :-1]
    mask_tgt = raw["mask"][:, 1:]
    mask = mask_src * mask_tgt
    valid_tokens = mask.sum() + 1e-6

    v_target = raw["dt_to_pre"][:, 1:]
    amt_target = raw["amt_to_pre"][:, 1:]

    sw_joint = feature_util.combine_sw({
        "sw_ip": raw["sw_ip"][:, 1:],
        "sw_email": raw["sw_email"][:, 1:],
        "sw_fp": raw["sw_fp"][:, 1:],
        "sw_bill": raw["sw_bill"][:, 1:],
        "sw_ship": raw["sw_ship"][:, 1:],
    })

    l_v = F.smooth_l1_loss(
        preds["v_pred"],
        v_target,
        reduction="none",
    ) * mask

    l_amt = F.smooth_l1_loss(
        preds["amt_pred"],
        amt_target,
        reduction="none",
    ) * mask

    l_sw_raw = F.cross_entropy(
        preds["sw_logits"].transpose(1, 2),
        sw_joint,
        reduction="none",
    )
    l_sw = feature_util.apply_sw_smooth_weight(l_sw_raw * mask, None)

    is_new_tgt = feature_util.combine_is_new(raw)

    expected_is_new_classes = getattr(current_cfg, "is_new_classes", None)
    if expected_is_new_classes is not None and is_new_tgt.size(-1) != expected_is_new_classes:
        raise ValueError(
            "is_new target/model dimension mismatch: "
            f"target has {is_new_tgt.size(-1)} labels, "
            f"cfg.is_new_classes={expected_is_new_classes}."
        )

    l_is_new_raw = F.binary_cross_entropy_with_logits(
        preds["is_new_logits"],
        is_new_tgt,
        reduction="none",
    )
    l_is_new = l_is_new_raw.sum(dim=-1) * mask

    # Unweighted component losses.
    loss_v = l_v.sum() / valid_tokens
    loss_sw = l_sw.sum() / valid_tokens
    loss_amt = l_amt.sum() / valid_tokens
    loss_is_new = l_is_new.sum() / valid_tokens

    total = (
        current_cfg.lambda_v * loss_v
        + current_cfg.lambda_sw * loss_sw
        + current_cfg.lambda_amt * loss_amt
        + current_cfg.lambda_is_new * loss_is_new
    )

    sw_ce = ((l_sw_raw * mask).sum() / valid_tokens).item()

    loss_components = {
        "v": loss_v.item(),
        "sw": loss_sw.item(),
        "amt": loss_amt.item(),
        "is_new": loss_is_new.item(),
        "sw_ce": sw_ce,
    }

    return total, loss_components


def build_txn_index(data) -> Dict[int, List[int]]:
    cid = data["trx_id"].numpy()
    groups: Dict[int, List[int]] = {}
    for i, c in enumerate(cid.tolist()):
        groups.setdefault(c, []).append(i)
    return groups


def _stage_probe_config() -> StageProbeConfig:
    return StageProbeConfig(
        enabled=True,
        activation=True,
        gradient=True,
        parameter=False,
        update=False,
        backward_flow=False,
        capture_input_activation=True,
        capture_output_activation=True,
    )


def build_diagnostic_probe(
    current_cfg,
    current_model,
    config_path=None,
) -> DiagnosticProbeManager:
    """Build diagnostics directly from the DAG-built model.

    ``model_structure.yaml`` is now the model-structure source of truth. The old
    rendered ``model_structure.diagnostic.json`` is intentionally not loaded or
    required. ``config_path`` is accepted only as a temporary call-site
    compatibility argument and is ignored.
    """
    del config_path

    from model_diagnostic.dag.model_builder_registry import iter_trainable_model_nodes

    DIAGNOSTICS.clear(reset_specs=True)
    probe = DiagnosticProbeManager(model=current_model, registry=DIAGNOSTICS)

    runtime_modules = []
    for node_id, module in iter_trainable_model_nodes(current_model):
        probe.register_stage(
            node_id,
            module,
            config=_stage_probe_config(),
        )
        runtime_modules.append({
            "module_path": f"model_nodes.{node_id}",
            "class_name": type(module).__name__,
            "diagnostics": {
                "enabled": True,
                "activation": True,
                "gradient": True,
                "parameter": False,
                "update": False,
                "backward_flow": False,
            },
        })

    # Temporary prediction helper remains outside model_structure.yaml, but it
    # still participates in optimization/diagnostics until the next DAG lands.
    prediction_helper = getattr(current_model, "prediction_helper", None)
    if prediction_helper is not None:
        for head_name, module in prediction_helper.named_children():
            stage = f"prediction.{head_name}"
            probe.register_stage(
                stage,
                module,
                config=_stage_probe_config(),
            )
            runtime_modules.append({
                "module_path": f"prediction_helper.{head_name}",
                "class_name": type(module).__name__,
                "diagnostics": {
                    "enabled": True,
                    "activation": True,
                    "gradient": True,
                    "parameter": False,
                    "update": False,
                    "backward_flow": False,
                },
            })

    # Keep an in-memory runtime description for diagnostic-runtime callers that
    # inspect probe.model_structure. It is derived from the DAG-built model and
    # is not a generated structure/config artifact.
    probe.model_structure = {
        "artifact_type": "dag_model_runtime_structure",
        "dag_id": "model_structure",
        "structure_id": "model_structure.yaml",
        "modules": runtime_modules,
    }

    probe.start()
    enabled = probe.list_stages()
    print(
        f"Diagnostics enabled from model DAG for {len(enabled)} module(s): "
        + ", ".join(enabled.keys())
    )
    return probe


def train_internal(
    current_cfg,
    current_model,
    probe: Optional[DiagnosticProbeManager] = None,
):
    print("Generating anchored training data (gen_anchored_dataset)...")
    train_data = seq_gen.make_anchored_dataset(
        current_cfg,
        num_trx=current_cfg.num_trx,
        is_training=True,
    )
    n_records = len(train_data["dt"])
    print(
        f"  records: {n_records}  "
        "(main+contrast views across cross-anchor pairs)"
    )

    txn_index = build_txn_index(train_data)
    print(
        f"  transactions: {len(txn_index)}  "
        "(sampled per step for batch diversity)"
    )

    opt = torch.optim.Adam(current_model.parameters(), lr=current_cfg.lr)
    n_params = sum(p.numel() for p in current_model.parameters())

    print(
        "Training Seq-on-Graph "
        f"(fwd-only residual GRU, hidden={current_cfg.hidden_dim}, "
        f"params={n_params:,}, "
        f"steps={current_cfg.steps}, "
        f"txn_batch={current_cfg.txn_batch_size})"
    )

    current_model.train()
    for step in range(current_cfg.steps):
        step_context = (
            probe.step(step, phase="train")
            if probe is not None
            else nullcontext()
        )

        with step_context:
            opt.zero_grad(set_to_none=True)

            batch, _, _ = batch_util.sample_txn_batch(
                train_data,
                txn_index,
                current_cfg.txn_batch_size,
            )
            x_full, raw = build_full_features(
                batch,
                current_cfg,
                is_training=True,
            )

            fwd_out = current_model(x_full)
            predicts = current_model.prediction_helper(fwd_out)

            l_recon, loss_components = compute_nextstep_recon_loss(
                predicts,
                raw,
                current_cfg,
            )

            loss = current_cfg.lambda_recon * l_recon

            if probe is not None:
                probe.store_many(
                    "training",
                    {
                        "loss.total": loss.item(),
                        "loss.recon": l_recon.item(),
                        "loss.v": loss_components["v"],
                        "loss.sw": loss_components["sw"],
                        "loss.amt": loss_components["amt"],
                        "loss.is_new": loss_components["is_new"],
                        "loss.sw_ce": loss_components["sw_ce"],
                        "optimizer.lr": opt.param_groups[0]["lr"],
                    },
                )

            loss.backward()

            # Collect raw stage gradients BEFORE clipping so the diagnostic
            # reflects the actual backward signal generated by this batch.
            if probe is not None:
                probe.after_backward()

            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                current_model.parameters(),
                current_cfg.clip_val,
            )

            if probe is not None:
                grad_norm_value = float(grad_norm_pre_clip)
                probe.store_many(
                    "training",
                    {
                        "grad.global.l2_pre_clip": grad_norm_value,
                        "grad.clip.max_norm": float(current_cfg.clip_val),
                        "grad.clip.triggered": grad_norm_value > current_cfg.clip_val,
                    },
                )

                # No snapshots are created unless a stage has update=true.
                probe.before_optimizer_step()

            opt.step()

            if probe is not None:
                probe.after_optimizer_step()

            if step % current_cfg.print_every == 0:
                print(
                    f"Step {step:04d} | Loss {loss.item():.4f} "
                    f"(Recon {l_recon.item():.4f} "
                    f"| V {loss_components['v']:.4f} "
                    f"| SW {loss_components['sw']:.4f} "
                    f"| AMT {loss_components['amt']:.4f} "
                    f"| NEW {loss_components['is_new']:.4f} "
                    f"| SW_CE {loss_components['sw_ce']:.4f})"
                )

    return current_model


def train(
    current_cfg,
    current_model,
    *,
    probe: Optional[DiagnosticProbeManager] = None,
):
    set_seed(current_cfg.seed)

    owns_probe = probe is None
    if probe is None:
        probe = build_diagnostic_probe(current_cfg, current_model)

    try:
        return train_internal(current_cfg, current_model, probe=probe)
    finally:
        if owns_probe and probe is not None:
            probe.stop()


def init_model(current_cfg):
    """Build the encoder from model_structure.yaml using the generic DAG engine."""
    processor, model_cfg = _get_model_dag_runtime()

    from model_diagnostic.dag_model_ops import (
        NextStepPredictionHelper,
        symbolic_input,
    )

    model = processor.run(
        model_cfg,
        inputs={"x_full": symbolic_input("x_full")},
        runtime={"cfg": current_cfg},
    )["model"]

    # Transitional seam only. It is deliberately visible here instead of being
    # hidden in model_structure.yaml. Remove it when prediction + loss become a
    # separate DAG. Registering it as a child keeps optimizer/state_dict behavior
    # equivalent to the current training pipeline during the migration.
    model.prediction_helper = NextStepPredictionHelper(
        hidden_dim=current_cfg.hidden_dim,
        sw_classes=current_cfg.sw_classes,
        is_new_classes=current_cfg.is_new_classes,
    )
    model.model_structure_path = str(model_cfg)

    return model.to(current_cfg.device)

def load_trained_model(current_cfg):
    model = init_model(current_cfg)
    model.load_state_dict(
        torch.load(
            current_cfg.model_path,
            map_location=current_cfg.device,
        )
    )
    model.eval()
    print(
        "Loaded DAG-built Seq-on-Graph model "
        f"(fwd-only residual GRU, hidden={current_cfg.hidden_dim})"
    )
    return model


def save_model(current_cfg, current_model):
    model_dir = os.path.dirname(current_cfg.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(current_model.state_dict(), current_cfg.model_path)
    print(f"Saved: {current_cfg.model_path}")


if __name__ == "__main__":

    should_init_model = True
    cfg = CFG2()
    set_seed(cfg.seed)

    model = init_model(cfg) if should_init_model else load_trained_model(cfg)
    train(cfg, model)
    save_model(cfg, model)
