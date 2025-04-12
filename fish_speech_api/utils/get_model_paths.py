import os


def get_model_paths(model_name: str) -> dict:
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found: {model_name}")
    return {"path": model_dir}
