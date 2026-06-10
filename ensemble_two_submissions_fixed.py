import os
import numpy as np
import pandas as pd

# =========================================================
# 改這裡：兩份 submission csv
# A 建議放目前最好的那份，例如 0.8041
# B 放另一份，例如 PatchTST drought 0.8271
# =========================================================
CSV_A = "/mnt/nas/users/sien/data_final/Experiments/ensemble_N-hits_40_PatchTST_60.csv"
CSV_B = "/mnt/nas/users/sien/data_final/Experiments/submission_tcn_k7_drought_perweek_w65_60_60_55_55.csv"  

OUT_DIR = "./ensemble_two_csv_outputs_nhits_PatchTST__tcn"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "region_id"
PRED_COLS = [
    "pred_week1",
    "pred_week2",
    "pred_week3",
    "pred_week4",
    "pred_week5",
]

# A 的權重；B 的權重會自動是 1 - A
WEIGHTS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# =========================================================
# 讀檔
# =========================================================
a = pd.read_csv(CSV_A)
b = pd.read_csv(CSV_B)

a[ID_COL] = a[ID_COL].astype(str)
b[ID_COL] = b[ID_COL].astype(str)

for c in PRED_COLS:
    if c not in a.columns:
        raise ValueError(f"CSV_A 缺少欄位: {c}")
    if c not in b.columns:
        raise ValueError(f"CSV_B 缺少欄位: {c}")

# 確保 region_id 順序一致
if not a[ID_COL].equals(b[ID_COL]):
    print("region_id 順序不同，依照 CSV_A 順序 merge CSV_B")
    b = a[[ID_COL]].merge(b, on=ID_COL, how="left")

    if b[PRED_COLS].isna().any().any():
        raise ValueError("merge 後有 NaN，代表兩份 csv 的 region_id 不完全一致")

pred_a = a[PRED_COLS].values.astype(np.float32)
pred_b = b[PRED_COLS].values.astype(np.float32)

print("CSV_A:", CSV_A)
print("CSV_B:", CSV_B)
print("Shape:", pred_a.shape)

print("\n=== CSV_A describe ===")
print(a[PRED_COLS].describe())

print("\n=== CSV_B describe ===")
print(b[PRED_COLS].describe())


# =========================================================
# 固定比例 ensemble
# =========================================================
summary_rows = []

for w in WEIGHTS:
    final = w * pred_a + (1.0 - w) * pred_b
    final = np.clip(final, 0, 5)

    out = a.copy()
    out[PRED_COLS] = final

    out_name = f"ensemble_A{int(round(w * 100))}_B{int(round((1.0 - w) * 100))}.csv"
    out_path = os.path.join(OUT_DIR, out_name)
    out.to_csv(out_path, index=False)

    summary_rows.append({
        "file": out_name,
        "A_weight": w,
        "B_weight": 1.0 - w,
        "mean_week1": final[:, 0].mean(),
        "mean_week2": final[:, 1].mean(),
        "mean_week3": final[:, 2].mean(),
        "mean_week4": final[:, 3].mean(),
        "mean_week5": final[:, 4].mean(),
        "std_all": final.std(),
        "min_all": final.min(),
        "max_all": final.max(),
    })

    print("=" * 80)
    print("Saved:", out_path)
    print(f"A weight = {w:.2f}, B weight = {1.0 - w:.2f}")
    print(out[PRED_COLS].describe())


# =========================================================
# 儲存 summary
# =========================================================
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "ensemble_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nAll ensemble files saved to:", OUT_DIR)
print("Summary saved to:", summary_path)