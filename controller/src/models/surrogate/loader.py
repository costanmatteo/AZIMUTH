import torch


def load_casualit_model(checkpoint_path: str, device: str):
    """
    Load a trained CausalIT model from checkpoint.

    Loads PyTorch Lightning checkpoints produced by the causaliT training
    pipeline (trainer.save_checkpoint).

    Args:
        checkpoint_path: Path to .ckpt file
        device: Torch device

    Returns:
        Tuple of (model, model_type)
    """
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint_data.get('model_type', 'proT')

    is_pl_checkpoint = 'pytorch-lightning_version' in checkpoint_data

    # CausaliT Lightning forecaster classes
    if model_type == 'proT':
        from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
        forecaster_cls = TransformerForecaster
    elif model_type == 'StageCausaliT':
        from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
        forecaster_cls = StageCausalForecaster
    elif model_type == 'SingleCausalLayer':
        from causaliT.training.forecasters.single_causal_forecaster import SingleCausalForecaster
        forecaster_cls = SingleCausalForecaster
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in checkpoint. "
            f"Expected 'proT', 'StageCausaliT', or 'SingleCausalLayer'.")

    if is_pl_checkpoint:
        model = forecaster_cls.load_from_checkpoint(
            checkpoint_path, map_location=device, weights_only=False)
    else:
        # Manual load from non-PL checkpoint with CausaliT config
        hparams = checkpoint_data.get('hyper_parameters', checkpoint_data.get('hparams', {}))
        config = hparams if hparams else checkpoint_data.get('config', {})
        if not config:
            raise ValueError(
                f"Cannot find hyperparameters/config in checkpoint {checkpoint_path}. "
                f"Available keys: {list(checkpoint_data.keys())}")
        state_dict = checkpoint_data.get('state_dict', checkpoint_data)
        model = forecaster_cls(config)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    # NOTE: Do NOT use requires_grad_(False) here. Freezing parameters
    # blocks gradient flow through the model (including w.r.t. inputs).
    # The surrogate parameters are not in the controller's optimizer,
    # so they won't be updated. But we need the computation graph intact
    # for gradients to flow from F back to the controller.
    model.to(device)
    print(f"  CasualiTSurrogate loaded model_type='{model_type}' from {checkpoint_path}")

    return model, model_type
