import math
import torch.nn as nn

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
