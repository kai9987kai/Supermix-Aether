"""Validated model cloning without mutable NumPy topology aliases."""

from aether.kernel.runtime import TissueModel


def clone_model(model: TissueModel) -> TissueModel:
    """Reconstruct an independent CPU model and validate every loaded buffer."""
    if not isinstance(model, TissueModel):
        raise ValueError('live models must be TissueModel instances')
    clone = TissueModel(model.config, model.protocol, model.seed)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.train(model.training)
    return clone
