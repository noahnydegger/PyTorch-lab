"""Training and evaluation steps for the lab's regression models."""

import torch  # Provides no_grad for evaluation.
from torch.nn.functional import mse_loss  # Computes mean squared error.


def common_step(model, features, targets):
    """Predict one value per sample and compute mean squared error."""
    # Calling the model runs its forward method and returns shape (samples,).
    predictions = model(features)
    # Both vectors must contain one value per sample. For example, allowing
    # (3, 1) and (3,) would broadcast them to (3, 3) and compare all pairs.
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")
    # MSE squares each prediction minus its target and averages the squares.
    # The result is a scalar tensor of shape (). In training it retains the
    # connection to the model parameters that backward() uses.
    loss = mse_loss(predictions, targets)
    return predictions, loss


def training_step(model, features, targets, optimizer):
    """Update the parameters once and return the loss before the update."""
    # Set training mode for layers that behave differently during training.
    # This linear layer behaves the same in both modes.
    model.train()
    # Clear stored gradients so this update uses only the current batch.
    optimizer.zero_grad()
    # Keep the differentiable loss. The underscore marks the unused predictions.
    _, loss = common_step(model, features, targets)
    # Backpropagate through the loss and store a gradient on each parameter.
    loss.backward()
    # Use those gradients to change the parameters. With SGD in this lesson,
    # each parameter moves by minus the learning rate times its gradient.
    optimizer.step()
    # Convert the already-computed loss to a Python number for logging.
    # This is the pre-update loss. To get the new loss, run another forward pass.
    return loss.item()


# Disable gradient recording while this function runs and restore the caller's
# gradient setting when it returns. The forward pass and MSE still run.
@torch.no_grad()
def validation_step(model, features, targets):
    """Score predictions with the current parameters."""
    # Set evaluation mode for layers such as dropout that check this flag.
    # The flag and the no_grad decorator control different things.
    model.eval()
    # Return one prediction per sample and a scalar loss. The observed targets
    # are used to calculate the score. There is no parameter update here.
    return common_step(model, features, targets)
