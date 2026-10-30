from types import SimpleNamespace
import pytest
import torch
from supermimic.algo.state import StateExtractor, encoder_bias_from_context


def test_bias_slice_matches_actual_context_producer_without_com_leak():
    extractor = object.__new__(StateExtractor)
    expected = torch.arange(30, dtype=torch.float32).reshape(1, 30) / 100
    extractor.robot = SimpleNamespace(data=SimpleNamespace(encoder_bias=expected))
    model = SimpleNamespace(
        geom_friction=torch.tensor([[[0.6, 0.0, 0.0], [0.8, 0.0, 0.0]]]),
        body_ipos=torch.tensor([[[10.0, 20.0, 30.0]]]),
    )
    extractor.env = SimpleNamespace(sim=SimpleNamespace(model=model))
    extractor.left_foot_geom_ids = torch.tensor([0])
    extractor.right_foot_geom_ids = torch.tensor([1])
    extractor.torso_body_id = 0
    extractor.default_body_ipos = torch.zeros((1, 3))
    context = extractor.context()
    assert context.shape == (1, 34)
    torch.testing.assert_close(context[:, 0], torch.tensor([0.7]))
    torch.testing.assert_close(encoder_bias_from_context(context, 30), expected)
    assert not torch.equal(
        context[:, 2:32], expected
    )  # Previous upstream offset included COM x.


def test_bias_slice_rejects_wrong_context_contract():
    with pytest.raises(ValueError, match="context width"):
        encoder_bias_from_context(torch.zeros(2, 35), 30)
