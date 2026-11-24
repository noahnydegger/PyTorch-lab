# PyTorch Lab

Learn PyTorch in small exercises. Predict the shape of each result, run the code, and explain what happened in your own words.

## Setup

From the repository root on macOS or Linux, create a virtual environment if you need one, then install the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Lesson 1 · Tensor shapes

1. Launch Jupyter with `python -m notebook 01_tensor_shapes.ipynb`. Run one cell at a time to read a synthetic discharge hydrograph, select a gauge, handle a missing measurement, and estimate volume above a flow threshold.
2. Run `python 01_tensor_shapes.py` for a terminal walkthrough. Edit the input tensor or try the prompts at the bottom, then rerun it to see what changes.

## Lesson 2 · Supervised learning and discharge forecasts

Open `02_discharge_baselines.ipynb` to turn a discharge series into next-day forecasts. Try each short task before expanding its solution. Run `python 02_discharge_baselines.py` to explore how changing the test flows affects the scores.

## Lesson 3 · Autograd and the training loop

Open `03_autograd.ipynb` to calculate one gradient and parameter update by hand, then reproduce it with PyTorch. The notebook imports the regression model and training steps from `pytorch_lab/` and connects those files to DROP's `RunoffNet`.

## Lesson 4 · Feed-forward networks for local runoff

Open `04_feedforward_runoff.ipynb` to work through ReLU by hand, train a two-catchment runoff model, and trace how its encoder and prediction head relate to DROP. The model lives in `pytorch_lab/nn/runoff_mlp.py` and uses the training functions from Lesson 3. The notebook ends with suggested reading from the Hands-On Deep Learning lecture.

## Later · Discharge observations

We'll use the floodwater-harvesting project's existing loader to read its processed discharge observations. We'll inspect the returned data and repeat the shape and indexing exercises without rerunning preprocessing.
