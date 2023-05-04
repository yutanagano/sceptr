from importlib.resources import files
import json
import torch


# Main model
MAIN_MODEL_NAME = "CDRBERT_+ACL_cns_020_cdrcns_001_chaincns_020"

with (files(__name__) / MAIN_MODEL_NAME / "config.json").open("rb") as f:
    BCDRBERT_ACL_CNS_CONFIG = json.load(f)
with (files(__name__) / MAIN_MODEL_NAME / "state_dict.pt").open("rb") as f:
    BCDRBERT_ACL_CNS_SD = torch.load(f)
