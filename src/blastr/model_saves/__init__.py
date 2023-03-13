import json
from pkg_resources import resource_stream
import torch


# BCDRBERT +ACL (cns)

with resource_stream(__name__, 'BCDRBERT_+ACL_cns/config.json') as f:
    BCDRBERT_ACL_CNS_CONFIG = json.load(f)
with resource_stream(__name__, 'BCDRBERT_+ACL_cns/state_dict.pt') as f:
    BCDRBERT_ACL_CNS_SD = torch.load(f)