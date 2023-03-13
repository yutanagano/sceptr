"""
default functions/methods
"""


from .model_saves import *
from .model_wrapper import ModelWrapper
import numpy as np
from pandas import DataFrame


default_model = ModelWrapper(
    config=BCDRBERT_ACL_CNS_CONFIG,
    state_dict=BCDRBERT_ACL_CNS_SD
)


def embed(data: DataFrame) -> np.ndarray:
    return default_model.embed(data)


def cdist(data_a: DataFrame, data_b: DataFrame) -> np.ndarray:
    return default_model.cdist(data_a, data_b)


def pdist(data: DataFrame) -> np.ndarray:
    return default_model.pdist(data)