import torch

from xvla.models.configuration_xvla import XVLAConfig
from xvla.models.transformer import SoftPromptedTransformer, timestep_embedding


def _model(enabled: bool) -> SoftPromptedTransformer:
    return SoftPromptedTransformer(
        hidden_size=16,
        multi_modal_input_size=8,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        num_domains=2,
        dim_action=4,
        dim_propio=4,
        dim_time=4,
        len_soft_prompts=0,
        max_len_seq=64,
        use_hetero_proj=False,
        use_main_visual_projection=enabled,
    )


def _inputs() -> dict:
    return {
        "domain_id": torch.tensor([0, 1]),
        "vlm_features": torch.randn(2, 3, 8),
        "aux_visual_inputs": torch.randn(2, 4, 8),
        "action_with_noise": torch.randn(2, 5, 4),
        "proprio": torch.randn(2, 4),
        "t": torch.rand(2),
    }


def test_configuration_defaults_to_historical_projection_layout():
    assert XVLAConfig().use_main_visual_projection is False


def test_disabled_projection_is_exactly_the_historical_forward():
    torch.manual_seed(7)
    model = _model(False).eval()
    inputs = _inputs()
    assert model.main_visual_proj is None

    with torch.no_grad():
        actual = model(**inputs)

        batch_size, num_actions = inputs["action_with_noise"].shape[:2]
        time_emb = timestep_embedding(inputs["t"], model.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(
            batch_size, num_actions, model.dim_time
        )
        proprio_tokens = inputs["proprio"].unsqueeze(1).expand(
            batch_size, num_actions, inputs["proprio"].shape[-1]
        )
        action_tokens = torch.cat(
            [inputs["action_with_noise"], proprio_tokens, time_tokens], dim=-1
        )
        hidden = model.action_encoder(action_tokens, inputs["domain_id"])
        hidden = torch.cat(
            [
                hidden,
                model.vlm_proj(inputs["vlm_features"]),
                model.aux_visual_proj(inputs["aux_visual_inputs"]),
            ],
            dim=1,
        )
        hidden = hidden + model.pos_emb[:, : hidden.shape[1], :]
        for block in model.blocks:
            hidden = block(hidden)
        expected = model.action_decoder(
            model.norm(hidden[:, :num_actions]), inputs["domain_id"]
        )

    assert torch.equal(actual, expected)


def test_enabled_projection_requires_and_consumes_main_tokens():
    model = _model(True).eval()
    inputs = _inputs()
    assert model.main_visual_proj is not None
    assert model.main_visual_proj.weight is not model.aux_visual_proj.weight

    try:
        model(**inputs)
    except ValueError as exc:
        assert "main_visual_inputs is required" in str(exc)
    else:
        raise AssertionError("enabled projection accepted missing main-camera tokens")

    with torch.no_grad():
        output = model(**inputs, main_visual_inputs=torch.randn(2, 2, 8))
    assert output.shape == (2, 5, 4)
