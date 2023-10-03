from importlib import resources
import json
import torch


class ModelSave:
    def __init__(self, config: dict, state_dict: dict) -> None:
        self.config = config
        self.state_dict = state_dict


def load_model_save(model_name: str) -> ModelSave:
    model_save = resources.files(__name__) / model_name

    with (model_save / "config.json").open("r") as f:
        config = json.load(f)

    with (model_save / "state_dict.pt").open("rb") as f:
        state_dict = torch.load(f)

    return ModelSave(config, state_dict)


beta_cdr_bert_unsupervised_model_save = load_model_save("Beta_CDR_BERT_Unsupervised")
