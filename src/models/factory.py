from src.models.base import ForecastModel, count_parameters, get_model_size

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(model_class):
        if not issubclass(model_class, ForecastModel):
            raise TypeError("Registered models must inherit from ForecastModel")
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        MODEL_REGISTRY[name] = model_class
        return model_class
    return decorator


def create_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown model: {name}. Available models: {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    return sorted(MODEL_REGISTRY)
