
from src.models.base import ForecastModel, count_parameters, get_model_size
from src.models.factory import create_model, list_models, register_model
from src.models.itransformer import ITransformer
from src.models.mamba_ts import MambaTS
from src.models.patchtst import PatchTST

register_model("itransformer")(ITransformer)
register_model("mambats")(MambaTS)
register_model("patchtst")(PatchTST)

__all__ = [
    "ForecastModel",
    "count_parameters",
    "get_model_size",
    "create_model",
    "list_models",
    "register_model",
    "ITransformer",
    "MambaTS",
    "PatchTST",
]
