"""Runs a reference trainer module unchanged and records every training state's loss terms per head, in the format of the losses.jsonl
of nopeworld.training.trainer, so the two can be compared state by state.

    NOPE_REFERENCE_TRAINER=<module with Head and main()> NOPE_REFERENCE_LOSSES=<out.jsonl> python tests/gpu/reference_trainer_losses.py <trainer arguments>

The reference code must be on PYTHONPATH. Only calls made with gradients enabled are recorded (the validation's are not).
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import torch


def main():
    module = importlib.import_module(os.environ["NOPE_REFERENCE_TRAINER"])
    out = open(os.environ["NOPE_REFERENCE_LOSSES"], "a")
    original = module.Head.losses

    def losses(self, model_out, batch):
        result = original(self, model_out, batch)
        if torch.is_grad_enabled():
            out.write(json.dumps({"head": self.name, "step": self.steps, "pending": self.pending,
                                  **{k: float(v.detach()) for k, v in result.items() if torch.is_tensor(v)}}) + "\n")
            out.flush()
        return result

    module.Head.losses = losses
    sys.argv = [module.__name__] + sys.argv[1:]
    module.main()


if __name__ == "__main__":
    main()
