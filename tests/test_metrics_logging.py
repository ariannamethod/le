import math
import torch.nn as nn
import pytest

import metrics
from le import CharDataset


def test_log_loss_and_perplexity():
    metrics.reset_metrics()
    metrics.set_writer(None)
    metrics.log_loss("train", 0.5, 1)
    assert metrics.get_metric("Loss/train") == 0.5
    assert metrics.get_metric("Perplexity/train") == math.exp(0.5)


def test_resonance_logging():
    metrics.reset_metrics()
    metrics.set_writer(None)

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(10, 4)

    dataset = CharDataset(["ab"], list("ab"), 2)
    model = Dummy()
    res = metrics.compute_resonance(model, dataset, "a", "b")
    metrics.log_resonance(res, 0)
    assert metrics.get_metric("Resonance") == res
    assert -1.0 <= res <= 1.0


def test_unique_token_ratio():
    metrics.reset_metrics()
    metrics.log_unique_token_ratio("a a b", 0)
    assert metrics.get_metric("UniqueTokenRatio") == pytest.approx(2 / 3)


def test_avg_response_length():
    metrics.reset_metrics()
    metrics.log_avg_response_length("a b", 0)
    metrics.log_avg_response_length("c d e", 1)
    assert metrics.get_metric("AvgResponseLength") == pytest.approx(2.5)


def test_ngram_repeat_rate():
    metrics.reset_metrics()
    metrics.log_ngram_repeat_rate("a b a b", 0, n=2)
    assert metrics.get_metric("RepeatRate") == pytest.approx(1 / 3)
