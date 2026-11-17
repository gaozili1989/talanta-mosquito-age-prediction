"""Shared architecture for training and reading saved ANN features."""


def build_ann(n_features, n_outputs, hidden=(100, 100)):
    from torch import nn

    first, second = hidden
    return nn.Sequential(
        nn.Linear(n_features, first), nn.ReLU(),
        nn.Linear(first, second), nn.ReLU(),
        nn.Linear(second, n_outputs),
    )
