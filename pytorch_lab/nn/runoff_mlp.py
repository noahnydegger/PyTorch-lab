"""Feed-forward runoff models with a reusable feature encoder."""

from torch import nn  # Provides trainable layers and activation functions.


class FeatureEncoder(nn.Module):
    """Turn the last feature axis into a learned representation."""

    def __init__(self, num_features, hidden_size):
        # Set up the module before assigning child layers.
        super().__init__()
        # Each hidden unit combines every feature using its own weights and bias.
        # ReLU follows the linear layer and clips negative values to zero.
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
        )

    def forward(self, features):
        # Apply the encoder to the final axis. Samples and any time axis stay
        # in place. For example, (batch, time, 2) → (batch, time, hidden_size).
        return self.layers(features)


class RunoffMLP(nn.Module):
    """Predict one local runoff value from each row of catchment features."""

    def __init__(self, num_features, hidden_size):
        # Register both the feature encoder and the output layer.
        super().__init__()
        self.encoder = FeatureEncoder(num_features, hidden_size)
        # The prediction head combines the hidden units into one scalar per row.
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features):
        # (samples, features) → (samples, hidden_size)
        encoded = self.encoder(features)
        # (samples, hidden_size) → (samples, 1)
        runoff = self.head(encoded)
        # Drop only the final axis. One sample still gives a vector of shape (1,).
        return runoff.squeeze(-1)
