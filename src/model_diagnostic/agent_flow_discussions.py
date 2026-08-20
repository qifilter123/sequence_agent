from __future__ import annotations

import json
from pathlib import Path

from cfg_base import CFG2
import encoder_model_train as train_driver
from diagnostic_registry import DIAGNOSTICS


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "model_structure.diagnostic.json"

# Agent pipeline example using python

def main() -> None:

    ## Step 1: collect static model structures

    '''
    Example building static model structures
    model = train_driver.init_model(cfg)
    static_structure = diagnostic_model_structure.render_model_structure(model)
    '''

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    ## Step 2: collect other information (architectures, config parameters...) used to configure model and train model.
    ### All the information collected will be sent as *system context* along with every runtime request to LLM

    ## Step 3: instructions and example how to run model, then start run model training

    cfg = CFG2()
    cfg.device = "cpu"

    '''model persist location'''
    cfg.model_path = str(HERE / "model" / "diagnostic_model.tar")
    cfg.diagnostic_config_path = str(CONFIG_PATH)
    '''init model'''
    model = train_driver.init_model(cfg)
    '''build  diagnostic prob'''
    probe = train_driver.build_diagnostic_probe(cfg, model, config_path=cfg.diagnostic_config_path)
    '''starts training'''
    cfg.num_trx = 80
    cfg.steps = 3
    cfg.txn_batch_size = 12
    cfg.print_every = 1
    try:
        train_driver.train_internal(cfg, model, probe=probe)
    finally:
        probe.stop()


    ## Step 4: Current training completes, ask LLM what metrics needed for this time.

    ## Step 5: fetch collected metrics

    loss_history = DIAGNOSTICS.get_history("training", "loss.total")
    clip = DIAGNOSTICS.get_current("training", "grad.clip.triggered")
    assert clip is not None

    ## Step 6: send metrics to LLM, request diagnosis from LLM

    ## Step 7: LL Agent either return back to Step 3 or continue

    ## Step 8: cleanups and save model and report
    '''save model'''
    train_driver.save_model(cfg, model)
    train_driver.load_trained_model(cfg)

    ### Generate report and persist

    '''report = ...
    REPORT_PATH = HERE / "e2e_validation_report.json"
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")'''

'''if __name__ == "__main__":
    main()'''
