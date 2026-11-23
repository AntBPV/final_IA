from typing import Any

import torch


def get_device(requested: str = "auto"):
    requested = requested.lower().strip()
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se solicito CUDA, pero PyTorch no fue instalado con soporte CUDA."
            )
        return torch.device("cuda")

    if requested == "dml":
        try:
            import torch_directml
        except ImportError as error:
            raise RuntimeError(
                "DirectML no esta disponible. Instale torch-directml en un "
                "entorno Python 3.10 compatible."
            ) from error
        return torch_directml.device()

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml

        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def device_summary(requested: str = "auto") -> dict[str, Any]:
    summary = {
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": torch.cuda.device_count(),
        "directml_available": False,
        "selected": None,
    }
    try:
        import torch_directml

        summary["directml_available"] = True
    except ImportError:
        pass

    try:
        summary["selected"] = str(get_device(requested))
    except RuntimeError as error:
        summary["selected"] = f"error: {error}"
    return summary
