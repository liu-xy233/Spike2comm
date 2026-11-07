"""Explicit ANN/SNN selection; old configs continue to select the ANN."""

from opencood.models.comm_modules.value_predictor import CounterfactualValuePredictor


def build_value_predictor(config):
    config = config.get("model", config.get("value_predictor", config))
    architecture = config.get("architecture", "ann")
    if architecture == "ann":
        return CounterfactualValuePredictor(config)
    if architecture == "sdtv3_snn":
        from opencood.models.comm_modules.spiking_value_predictor import SpikingCounterfactualValuePredictor
        return SpikingCounterfactualValuePredictor(config)
    raise ValueError("Unknown value predictor architecture: " + str(architecture))


def validate_value_predictor_checkpoint(model, checkpoint):
    """Check non-weight semantics (spikes, metadata scales, grid) before load.

strict=True alone cannot detect same-shaped models with different scales or
quantization. Legacy ANN checkpoints without config remain supported.
"""
    saved = checkpoint.get("config", {})
    saved = saved.get("model", saved.get("value_predictor", saved))
    expected = getattr(model, "architecture_config", None)
    if expected is not None:
        from opencood.models.comm_modules.spiking_value_predictor import resolve_spiking_config
        if saved.get("architecture") != "sdtv3_snn":
            raise ValueError("SNN checkpoint must include config.model.architecture=sdtv3_snn")
        actual = resolve_spiking_config(saved)
        mismatches = [key for key in expected if actual[key] != expected[key]]
        if mismatches:
            raise ValueError("SNN checkpoint configuration mismatch: " + ", ".join(mismatches))
    elif saved.get("architecture", "ann") != "ann":
        raise ValueError("Cannot load an SNN checkpoint into an ANN predictor")
