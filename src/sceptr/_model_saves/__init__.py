from importlib import resources
import json
from libtcrlm.config_reader import ConfigReader
import logging
import sceptr
from sceptr.model import Sceptr
import torch


logger = logging.getLogger(__name__)


def load_variant(model_name: str) -> Sceptr:
    logger.debug(f"Loading SCEPTR variant: {model_name}")

    model_save_dir = resources.files(__name__) / model_name

    with (model_save_dir / "config.json").open("r") as f:
        config = json.load(f)

    with (model_save_dir / "state_dict.pt").open("rb") as f:
        state_dict = torch.load(f, weights_only=True)

    config_reader = ConfigReader(config)

    name = config_reader.get_model_name()
    tokeniser = config_reader.get_tokeniser()
    bert = config_reader.get_bert()
    bert.load_state_dict(state_dict)

    model = Sceptr(name=name, tokeniser=tokeniser, bert=bert)

    if sceptr._USE_HARDWARE_ACCELERATION:
        model.enable_hardware_acceleration()

    return model
