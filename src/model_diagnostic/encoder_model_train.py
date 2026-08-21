from contextlib import nullcontext
from typing import Dict, List, Optional
from pathlib import Path
import os
import sys
import random

import numpy as np
import torch

from model_diagnostic import generic_seq_generator as seq_gen
from model_diagnostic import batch_util
from model_diagnostic.cfg_base import CFG2

from model_diagnostic.diagnostic_probe_manager import DiagnosticProbeManager, StageProbeConfig
from model_diagnostic.diagnostic_registry import DIAGNOSTICS


_FEATURE_DAG_RUNTIME = None


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

    prediction_loss = getattr(current_model, "prediction_loss", None)
    if prediction_loss is not None:
        for node_id, module in iter_trainable_model_nodes(prediction_loss):
            stage = f"prediction.{node_id}"
            probe.register_stage(
                stage,
                module,
                config=_stage_probe_config(),
            )
            runtime_modules.append({
                "module_path": f"prediction_loss.model_nodes.{node_id}",
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
            loss_out = current_model.prediction_loss(
                encoder_output=fwd_out,
                raw=raw,
            )

            loss = loss_out["loss"]
            l_recon = loss_out["recon_loss"]

            if probe is not None:
                probe.store_many(
                    "training",
                    {
                        "loss.total": loss.item(),
                        "loss.recon": l_recon.item(),
                        "loss.v": loss_out["loss_v"].item(),
                        "loss.sw": loss_out["loss_sw"].item(),
                        "loss.amt": loss_out["loss_amt"].item(),
                        "loss.is_new": loss_out["loss_is_new"].item(),
                        "loss.sw_ce": loss_out["sw_ce"].item(),
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
                    f"| V {loss_out['loss_v'].item():.4f} "
                    f"| SW {loss_out['loss_sw'].item():.4f} "
                    f"| AMT {loss_out['loss_amt'].item():.4f} "
                    f"| NEW {loss_out['loss_is_new'].item():.4f} "
                    f"| SW_CE {loss_out['sw_ce'].item():.4f})"
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
    """Build encoder + prediction/loss modules from their BUILD DAGs."""
    from model_diagnostic.dag.dag_processor import DagProcessor
    from model_diagnostic.dag.model_builder_registry import (
        MODEL_BUILDER_REGISTRY,
        symbolic_input,
    )

    # Import modules for their operation registrations. DagProcessor remains
    # generic and knows nothing about model or prediction/loss semantics.
    from model_diagnostic import dag_model_ops as _dag_model_ops
    from model_diagnostic import dag_prediction_loss_ops as _dag_prediction_loss_ops
    del _dag_model_ops, _dag_prediction_loss_ops

    config_dir = Path(__file__).resolve().parent / "config"
    model_config_path = config_dir / "model_structure.yaml"
    prediction_loss_config_path = config_dir / "prediction_loss.yaml"

    processor = DagProcessor(MODEL_BUILDER_REGISTRY)

    model_cfg = processor.load_config(model_config_path)
    model = processor.run(
        model_cfg,
        inputs={"x_full": symbolic_input("x_full")},
        runtime={"cfg": current_cfg},
    )["model"]

    prediction_loss_cfg = processor.load_config(prediction_loss_config_path)
    prediction_loss = processor.run(
        prediction_loss_cfg,
        inputs={
            "encoder_output": symbolic_input("encoder_output"),
            "raw": symbolic_input("raw"),
        },
        runtime={"cfg": current_cfg},
    )["model"]

    # Register as a child module so optimizer parameters, train/eval mode,
    # device moves, and state_dict all include prediction heads automatically.
    model.prediction_loss = prediction_loss
    model.model_structure_path = str(model_config_path)
    model.prediction_loss_path = str(prediction_loss_config_path)

    return model.to(current_cfg.device)

def _migrate_legacy_prediction_checkpoint_keys(state_dict):
    """Remap the transitional prediction_helper head keys to the DAG layout.

    The migration is intentionally load-only. New checkpoints are always saved
    with the prediction_loss.model_nodes.* structure.
    """
    migrated = False
    for head_name in ("v_head", "sw_head", "amt_head", "is_new_head"):
        for suffix in ("weight", "bias"):
            old_key = f"prediction_helper.{head_name}.{suffix}"
            new_key = f"prediction_loss.model_nodes.{head_name}.{suffix}"
            if old_key not in state_dict:
                continue
            if new_key in state_dict:
                raise RuntimeError(
                    "Checkpoint contains both legacy and DAG prediction-head keys: "
                    f"{old_key!r} and {new_key!r}"
                )
            state_dict[new_key] = state_dict.pop(old_key)
            migrated = True
    return migrated


def load_trained_model(current_cfg):
    model = init_model(current_cfg)
    state_dict = torch.load(
        current_cfg.model_path,
        map_location=current_cfg.device,
    )
    migrated = _migrate_legacy_prediction_checkpoint_keys(state_dict)
    model.load_state_dict(state_dict)
    if migrated:
        print("Migrated legacy prediction_helper checkpoint keys to prediction_loss DAG keys")
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
