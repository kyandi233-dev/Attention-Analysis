from __future__ import annotations

import sys

import torch

from export_ritnet_batch_variants import FinalUncertaintyExportWrapper, parse_args


class FakeModel(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        logits = torch.zeros((batch, 4, height, width), dtype=torch.float32)
        logits[:, 1] = 1.0
        logits[:, 3, :, : width // 2] = 2.0
        return logits


def test_final_uncertainty_wrapper_returns_pixelwise_four_class_probability():
    wrapper = FinalUncertaintyExportWrapper(FakeModel()).eval()
    image = torch.zeros((2, 1, 8, 12), dtype=torch.float32)
    labels, probability, maxprob, margin, entropy = wrapper(image)

    assert labels.shape == (2, 8, 12)
    assert probability.shape == (2, 4, 8, 12)
    assert maxprob.shape == (2, 8, 12)
    assert margin.shape == (2, 8, 12)
    assert entropy.shape == (2, 8, 12)
    assert torch.allclose(probability.sum(dim=1), torch.ones((2, 8, 12)), atol=1e-6, rtol=0)


def test_final_uncertainty_export_defaults_to_production_batch16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_ritnet_batch_variants.py", "--final-uncertainty"])
    args = parse_args()
    assert args.batches == "16"


def test_historical_export_default_batches_remain_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_ritnet_batch_variants.py"])
    args = parse_args()
    assert args.batches == "8,10,12,14"
