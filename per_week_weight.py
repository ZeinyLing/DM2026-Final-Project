import numpy as np
import pandas as pd

SAMPLE_SUBMISSION = "./data/sample_submission.csv"

model_preds = np.load("model_preds_drought_features.npy")
naive_preds = np.load("naive_preds_drought_features.npy")

sample = pd.read_csv(SAMPLE_SUBMISSION)
sample["region_id"] = sample["region_id"].astype(str)

PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]

weight_sets = {
    "w70_60_60_60_60": [0.70, 0.60, 0.60, 0.60, 0.60],
    "w70_65_60_55_50": [0.70, 0.65, 0.60, 0.55, 0.50],
    "w65_60_60_55_55": [0.65, 0.60, 0.60, 0.55, 0.55],
    "w65_60_55_55_50": [0.65, 0.60, 0.55, 0.55, 0.50],
    "w60_60_60_60_60": [0.60, 0.60, 0.60, 0.60, 0.60],
    "w60_55_55_50_50": [0.60, 0.55, 0.55, 0.50, 0.50],
}

for name, weights in weight_sets.items():
    weights = np.array(weights, dtype=np.float32)

    final_preds = np.zeros_like(model_preds)

    for i, w in enumerate(weights):
        final_preds[:, i] = w * model_preds[:, i] + (1.0 - w) * naive_preds[:, i]

    final_preds = np.clip(final_preds, 0, 5)

    out = sample.copy()

    for i, col in enumerate(PRED_COLS):
        out[col] = final_preds[:, i]

    out_name = f"submission_drought_perweek_{name}.csv"
    out.to_csv(out_name, index=False)

    print("=" * 80)
    print(f"Saved: {out_name}")
    print("weights:", weights.tolist())
    print(out[PRED_COLS].describe())