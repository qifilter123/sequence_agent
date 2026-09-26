import torch

from transformer.model_cfg import *
from transformer.model_transformer import StockTransformer


def save_model(model: StockTransformer) -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        MODEL_OUTPUT_PATH,
    )

def load_model(
    model_path: Path = MODEL_OUTPUT_PATH,
    device: torch.device | str | None = None,
) -> StockTransformer:
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    # First build the model using the current structure.
    model = StockTransformer().to(device)

    # Then load the trained parameters.
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()
    return model
