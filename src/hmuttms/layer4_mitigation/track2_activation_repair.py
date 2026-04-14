"""Track 2 — Activation-Based Repair (from GlitchProber).

For tokens that must be processed (blacklisted but still present),
register forward-hooks on the target model's MLP layers and patch the
intermediate activations so they match a "normal" reference profile.

Design
------
* A repair is stored in the Layer-3 ``metadata`` sub-DB under a key
  ``activation_profile`` (populated offline by GlitchProber-style probes
  over a reference corpus on Model B).
* The profile is a dict ``{layer_name: [target_activation_vector]}``; at
  runtime we subtract the token's observed activation delta toward the
  target, scaled by ``strength``.
* The track is implemented as a *context manager* so hooks are torn down
  after the forward pass even if it throws.

If no profile is present, the track declines and the pipeline escalates
to Track 3.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)


class Track2ActivationRepair(MitigationTrack):
    name = "track2_activation_repair"

    def __init__(self, strength: float = 0.5, layer_prefix: str = "mlp"):
        self.strength = float(strength)
        self.layer_prefix = layer_prefix

    # ---------------------------------------------------------------- applies
    def applies(self, context: MitigationContext, registry_view: dict) -> bool:
        meta = registry_view.get("metadata") or {}
        profile = meta.get("activation_profile")
        return bool(profile) and context.model is not None

    # ------------------------------------------------------------------ apply
    def apply(
        self, context: MitigationContext, registry_view: dict
    ) -> MitigationResult:
        meta = registry_view.get("metadata") or {}
        profile: Optional[dict] = meta.get("activation_profile")

        if profile is None:
            return MitigationResult(
                original_token_id=context.current_id,
                action=MitigationAction.REPAIR_ACTIVATION,
                track=self.name,
                success=False,
                reason="no_activation_profile",
            )

        # We don't run the forward pass here — the pipeline caller owns
        # generation. Instead, we stage the hooks under ``context.extras``
        # so the caller can install them via ``install_hooks_for(...)``.
        context.extras.setdefault("activation_patches", []).append(
            {
                "token_id": context.current_id,
                "position": context.position,
                "profile": profile,
                "strength": self.strength,
                "layer_prefix": self.layer_prefix,
            }
        )

        return MitigationResult(
            original_token_id=context.current_id,
            action=MitigationAction.REPAIR_ACTIVATION,
            track=self.name,
            activation_patches={
                "layer_prefix": self.layer_prefix,
                "strength": self.strength,
                "num_layers_patched": len(profile),
            },
            success=True,
            details={"staged": True},
        )


# --------------------------------------------------------------- hook helpers
@contextmanager
def install_hooks_for(
    model,
    patches: list[dict],
) -> Iterator[list]:
    """Install forward hooks on ``model`` that apply the staged patches.

    Use as::

        with install_hooks_for(model, context.extras["activation_patches"]) as handles:
            out = model.generate(...)

    Hooks remove themselves on exit even if ``generate()`` raises.
    """
    if not patches:
        yield []
        return

    import torch

    # Group patches by layer name for a single hook per module.
    by_layer: dict[str, list[dict]] = {}
    for patch in patches:
        for layer_name in patch["profile"].keys():
            by_layer.setdefault(layer_name, []).append(patch)

    handles = []
    named_modules = dict(model.named_modules())

    def _make_hook(layer_name: str, relevant: list[dict]):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return output
            for patch in relevant:
                pos = patch["position"]
                if pos >= output.shape[-2]:
                    continue
                target = torch.tensor(
                    patch["profile"][layer_name],
                    dtype=output.dtype,
                    device=output.device,
                )
                delta = target - output[..., pos, :]
                output[..., pos, :] = output[..., pos, :] + patch["strength"] * delta
            return output

        return hook

    try:
        for layer_name, relevant in by_layer.items():
            module = named_modules.get(layer_name)
            if module is None:
                continue
            handles.append(module.register_forward_hook(_make_hook(layer_name, relevant)))
        yield handles
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
