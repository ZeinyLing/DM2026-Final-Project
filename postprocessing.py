import pandas as pd
import numpy as np

ens = pd.read_csv("./ensemble_clean0.50_gap0.50.csv")

params = {
    "pred_week1": {"a": 0.7, "b": 0.5, "t": 0.55},
    "pred_week2": {"a": 0.8, "b": 0.4, "t": 0.53},
    "pred_week3": {"a": 0.8, "b": 0.3, "t": 0.75},
    "pred_week4": {"a": 0.8, "b": 0.3, "t": 0.81},
    "pred_week5": {"a": 0.8, "b": 0.3, "t": 0.76},
}

out = ens.copy()

for col, p in params.items():
    pred = out[col].values
    pred = p["a"] * pred + p["b"]
    pred = np.clip(pred, 0, 5)
    pred[pred < p["t"]] = 0
    out[col] = pred

out.to_csv("ensemble_v4.csv", index=False)
