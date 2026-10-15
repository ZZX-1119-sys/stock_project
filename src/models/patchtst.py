
import sys
from pathlib import Path
from types import SimpleNamespace

from src.models.base import ForecastModel

REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "_reference"
    / "PatchTST"
    / "PatchTST_supervised"
)





def _load_official_model():
    import importlib.util

    root = REFERENCE_ROOT.resolve()
    backbone_path = root / "models" / "PatchTST.py"

    package_name = "_official_patchtst"
    if package_name not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                package_name,
                root / "models" / "__init__.py",
                submodule_search_locations=[str(root / "models")],
            )
        )
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.PatchTST",
        backbone_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(root))
        layers_name = "_official_patchtst_layers"
        layers_spec = importlib.util.spec_from_file_location(
            layers_name,
            root / "layers" / "__init__.py",
            submodule_search_locations=[str(root / "layers")],
        )
        layers_module = importlib.util.module_from_spec(layers_spec)
        sys.modules[layers_name] = layers_module
        layers_module.__name__ = "layers"
        sys.modules["layers"] = layers_module

        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path

    return module.Model





class PatchTST(ForecastModel):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        target_dim,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
        patch_len=16,
        stride=8,
        revin=True,
        **kwargs,
    ):
        super().__init__(seq_len, pred_len, enc_in, target_dim)

        target_idx = kwargs.pop("target_idx", 3)
        if not 0 <= target_idx < enc_in:
            raise ValueError("target_idx must be within the input feature range")
        if target_dim != 1:
            raise ValueError("PatchTST currently supports one target feature")
        self.target_idx = target_idx

        if patch_len < 1 or patch_len > seq_len:
            raise ValueError("patch_len must be between 1 and seq_len")

        if stride < 1:
            raise ValueError("stride must be positive")

        configs = SimpleNamespace(
            enc_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=dropout,
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=revin,
            affine=True,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )

        official_model = _load_official_model()
        self.model = official_model(configs)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [Batch, Seq_Len, Features]"
            )

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[1]}"
            )

        if x.shape[2] != self.enc_in:
            raise ValueError(
                f"Expected {self.enc_in} input features, got {x.shape[2]}"
            )

        output = self.model(x)
        # The official PatchTST predicts every input channel. Keep the
        # requested target channel; close is index 3 in the project's
        # documented feature order.
        output = output[:, :, self.target_idx:self.target_idx + 1]

        if output.shape != (
            x.shape[0],
            self.pred_len,
            self.target_dim,
        ):
            raise RuntimeError(
                f"Unexpected output shape: {tuple(output.shape)}"
            )

        return output
