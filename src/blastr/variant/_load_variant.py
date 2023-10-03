from blastr._lib.blastr import Blastr
from blastr._lib.config_reader import ConfigReader
import torch


def load_variant(config: dict, state_dict: dict) -> Blastr:
    config_reader = ConfigReader(config)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    name = config_reader.get_model_name()
    tokeniser = config_reader.get_tokeniser()
    bert = config_reader.get_bert_on_device(device)
    bert.load_state_dict(state_dict)

    return Blastr(name=name, tokeniser=tokeniser, bert=bert, device=device)
