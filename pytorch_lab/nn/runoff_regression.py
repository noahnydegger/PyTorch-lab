"""A one-feature regression model for the discharge forecasting lesson."""

# nn provides the base class and the trainable linear layer used below.
from torch import nn


class RunoffRegression(nn.Module):
    """Map today's discharge to one next-day discharge per row."""

    def __init__(self):
        # Calling RunoffRegression() runs this method. Initialise nn.Module
        # first so it can register child layers and track their parameters.
        super().__init__()
        # One input and one output give a weight of shape (1, 1) and a bias
        # of shape (1,). The layer initialises both before any training.
        # Saving it as self.linear makes both available via model.parameters().
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, features):
        # model(features) calls this method. The layer applies weight × flow
        # + bias to each row, giving one output column of shape (samples, 1).
        predictions = self.linear(features)
        # Remove only the final axis to match the target shape (samples,).
        # With a single sample the result still has shape (1,).
        return predictions.squeeze(-1)
