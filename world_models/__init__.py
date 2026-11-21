"""World-model wrappers by model id: one adapter per third-party video world model."""
from __future__ import annotations

import importlib

WORLD_MODELS = {
    "lingbot_world_2": "lingbot_world_2:LingBotWorld2Adapter",
    "hy_worldplay": "hy_worldplay:HYWorldPlayAdapter",
    "echo_wm": "echo_wm:EchoWMAdapter",
    "evoke_turbo": "evoke_turbo:EvokeTurboAdapter",
    "worldplay2": "worldplay2:WorldPlay2Adapter",
    "hy_nomas": "hy_nomas:HYNomasAdapter",
}


def get_adapter(generator_id, config=None):
    if generator_id not in WORLD_MODELS:
        raise ValueError(f"unknown world model {generator_id!r}: one of {sorted(WORLD_MODELS)}")
    module, name = WORLD_MODELS[generator_id].split(":")
    cls = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    if config is None:
        from nopeworld.config import generator_config
        config = generator_config(generator_id)
    return cls(config)
