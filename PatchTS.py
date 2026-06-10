# =========================================================
# PatchTST Regression + Drought Cumulative Features + Naive Ensemble
# Natural Disaster Severity Prediction
# =========================================================

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")

SAVE_PATH = "best_patchtst_drought_features_patch14.pth"
OUT_CSV = "PatchTST_DroughtFeatures_patch14_dropout0.1_70_30.csv"

INPUT_LEN = 91
PRED_LEN = 5

PATCH_LEN = 14
STRIDE = 7

D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.1

BATCH_SIZE = 128
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.15
RECENT_RATIO = 0.5

ENSEMBLE_MODEL_WEIGHT = 0.7
ENSEMBLE_NAIVE_WEIGHT = 0.3

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# Drought cumulative feature engineering
# =========================================================
BASE_METEO_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
]


def add_drought_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    Add drought-related cumulative / rolling features.

    Important:
    - All features are computed within each region.
    - Rolling windows use current and past days only.
    - No target / future score leakage.
    """

    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    # Basic drought indicators
    if "prec" in df.columns:
        df["prec_log"] = np.log1p(np.clip(df["prec"].astype(np.float32), a_min=0, a_max=None))

    if "tmp" in df.columns and "humidity" in df.columns:
        # Higher value = hotter and drier
        humidity_stress = np.clip(100.0 - df["humidity"].astype(np.float32), a_min=0, a_max=None)
        df["hot_dry_index"] = df["tmp"].astype(np.float32) * humidity_stress

    if "tmp_max" in df.columns and "humidity" in df.columns:
        humidity_stress = np.clip(100.0 - df["humidity"].astype(np.float32), a_min=0, a_max=None)
        df["hot_dry_max_index"] = df["tmp_max"].astype(np.float32) * humidity_stress

    # Rolling cumulative / average features
    rolling_specs = []

    if "prec" in df.columns:
        for w in [7, 14, 28, 56]:
            rolling_specs.append(("prec", f"prec_roll{w}_sum", w, "sum"))

    if "humidity" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("humidity", f"humidity_roll{w}_mean", w, "mean"))

    if "tmp" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("tmp", f"tmp_roll{w}_mean", w, "mean"))

    if "tmp_max" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("tmp_max", f"tmpmax_roll{w}_mean", w, "mean"))

    if "wind" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("wind", f"wind_roll{w}_mean", w, "mean"))

    if "hot_dry_index" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("hot_dry_index", f"hotdry_roll{w}_mean", w, "mean"))

    if "hot_dry_max_index" in df.columns:
        for w in [7, 14, 28]:
            rolling_specs.append(("hot_dry_max_index", f"hotdrymax_roll{w}_mean", w, "mean"))

    # Efficient enough for this data size; may take a few minutes
    g = df.groupby("region_id", sort=False)

    for src_col, new_col, window, how in rolling_specs:
        if how == "sum":
            df[new_col] = g[src_col].transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            )
        elif how == "mean":
            df[new_col] = g[src_col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        else:
            raise ValueError(f"Unknown rolling type: {how}")

    # Precipitation deficit-like features
    # Difference between short-term and longer-term rainfall.
    if "prec_roll7_sum" in df.columns and "prec_roll28_sum" in df.columns:
        df["prec_roll7_minus_roll28_avg"] = (
            df["prec_roll7_sum"] - df["prec_roll28_sum"] / 4.0
        )

    if "prec_roll14_sum" in df.columns and "prec_roll56_sum" in df.columns:
        df["prec_roll14_minus_roll56_avg"] = (
            df["prec_roll14_sum"] - df["prec_roll56_sum"] / 4.0
        )

    # Temperature-humidity combined stress trends
    if "hotdry_roll7_mean" in df.columns and "hotdry_roll28_mean" in df.columns:
        df["hotdry_roll7_minus_roll28"] = (
            df["hotdry_roll7_mean"] - df["hotdry_roll28_mean"]
        )

    if "humidity_roll7_mean" in df.columns and "humidity_roll28_mean" in df.columns:
        df["humidity_roll7_minus_roll28"] = (
            df["humidity_roll7_mean"] - df["humidity_roll28_mean"]
        )

    if "tmp_roll7_mean" in df.columns and "tmp_roll28_mean" in df.columns:
        df["tmp_roll7_minus_roll28"] = (
            df["tmp_roll7_mean"] - df["tmp_roll28_mean"]
        )

    # Replace possible inf values
    for c in df.columns:
        if c not in ["region_id", "date"]:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df


# =========================================================
# DATASET
# =========================================================
class DisasterWindowDataset(Dataset):
    def __init__(self, region_data, samples):
        self.region_data = region_data
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, start, end, target_idx = self.samples[idx]

        feat = self.region_data[rid]["feat"]
        score = self.region_data[rid]["score"]

        x = feat[start:end]
        y = score[target_idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(rid, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32)
        )


class DisasterTestDataset(Dataset):
    def __init__(self, X, R):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.R[idx]


# =========================================================
# BUILD REGION DATA
# =========================================================
def build_region_data(train_df, feat_cols, region_map):
    region_data = {}

    train_df = train_df.sort_values(["region_id", "date"])

    for region, g in train_df.groupby("region_id"):
        g = g.sort_values("date").reset_index(drop=True)

        rid = region_map[region]

        region_data[rid] = {
            "region_name": region,
            "feat": g[feat_cols].values.astype(np.float32),
            "score": g["score"].values.astype(np.float32)
        }

    return region_data


# =========================================================
# BUILD TRAIN / VAL SAMPLES
# only recent windows
# =========================================================
def build_train_val_samples(region_data):
    train_samples = []
    val_samples = []

    for rid, data in region_data.items():
        score = data["score"]

        valid_idx = np.where(~np.isnan(score))[0]

        all_samples = []

        for i in range(len(valid_idx) - PRED_LEN + 1):
            target_idx = valid_idx[i:i + PRED_LEN]

            end = target_idx[0]
            start = end - INPUT_LEN

            if start < 0:
                continue

            y = score[target_idx]

            if np.any(np.isnan(y)):
                continue

            all_samples.append((rid, start, end, target_idx))

        if len(all_samples) == 0:
            continue

        use_start = int(len(all_samples) * (1 - RECENT_RATIO))
        recent_samples = all_samples[use_start:]

        split = int(len(recent_samples) * (1 - VAL_RATIO))

        train_samples.extend(recent_samples[:split])
        val_samples.extend(recent_samples[split:])

    return train_samples, val_samples


# =========================================================
# MODEL
# =========================================================
class PatchTST(nn.Module):
    def __init__(self, n_feat, n_region):
        super().__init__()

        self.patch_len = PATCH_LEN
        self.stride = STRIDE
        self.num_patches = ((INPUT_LEN - PATCH_LEN) // STRIDE) + 1

        self.proj = nn.Linear(PATCH_LEN, D_MODEL)

        self.pos_emb = nn.Parameter(
            torch.randn(1, n_feat, self.num_patches, D_MODEL)
        )

        self.region_emb = nn.Embedding(n_region, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.head = nn.Sequential(
            nn.LayerNorm(n_feat * self.num_patches * D_MODEL),
            nn.Linear(n_feat * self.num_patches * D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, PRED_LEN)
        )

    def forward(self, x, r):
        # x: [B, 91, C]
        B, T, C = x.shape

        x = x.permute(0, 2, 1)                         # [B, C, T]
        x = x.unfold(2, self.patch_len, self.stride)    # [B, C, P, patch_len]

        x = self.proj(x)                                # [B, C, P, D]
        x = x + self.pos_emb[:, :C, :, :]

        r = self.region_emb(r).view(B, 1, 1, -1)
        x = x + r

        x = x.reshape(B, C * self.num_patches, D_MODEL)

        x = self.encoder(x)

        x = x.reshape(B, -1)

        return self.head(x)


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for x, r, y in loader:
        x = x.to(DEVICE)
        r = r.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x, r)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0

    preds = []
    trues = []

    for x, r, y in loader:
        x = x.to(DEVICE)
        r = r.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x, r)
        loss = loss_fn(pred, y)

        pred_clamped = torch.clamp(pred, 0, 5)

        total_loss += loss.item() * x.size(0)

        preds.append(pred_clamped.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    week_mae = np.mean(np.abs(preds - trues), axis=0)

    return total_loss / len(loader.dataset), mae, rmse, week_mae


# =========================================================
# TEST DATA
# =========================================================
def build_test_samples(test_df, feat_cols, region_map, sample_df=None):
    test_df = test_df.sort_values(["region_id", "date"])

    X_list = []
    R_list = []
    region_names = []

    if sample_df is not None and "region_id" in sample_df.columns:
        order_regions = sample_df["region_id"].astype(str).tolist()
    else:
        order_regions = sorted(test_df["region_id"].astype(str).unique())

    grouped = {
        str(r): g.sort_values("date").reset_index(drop=True)
        for r, g in test_df.groupby("region_id")
    }

    for region in order_regions:
        region = str(region)

        if region not in grouped:
            raise ValueError(f"Region {region} not found in test.csv")

        g = grouped[region]
        feat = g[feat_cols].values.astype(np.float32)

        if len(feat) < INPUT_LEN:
            raise ValueError(f"Region {region} test length < {INPUT_LEN}")

        x = feat[-INPUT_LEN:]

        X_list.append(x)
        R_list.append(region_map[region])
        region_names.append(region)

    return np.array(X_list), np.array(R_list), region_names


# =========================================================
# NAIVE BASELINE
# use latest 5 non-NaN scores
# =========================================================
def build_naive_pred(train_df, region_names):
    naive_preds = []

    for region in region_names:
        g = train_df[train_df["region_id"] == str(region)].sort_values("date")
        scores = g["score"].dropna().values.astype(np.float32)

        if len(scores) >= PRED_LEN:
            pred = scores[-PRED_LEN:]
        elif len(scores) > 0:
            pred = np.full(PRED_LEN, scores[-1], dtype=np.float32)
        else:
            pred = np.zeros(PRED_LEN, dtype=np.float32)

        naive_preds.append(pred)

    return np.array(naive_preds, dtype=np.float32)


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_preds = []

    for x, r in loader:
        x = x.to(DEVICE)
        r = r.to(DEVICE)

        pred = model(x, r)
        pred = torch.clamp(pred, 0, 5)

        all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# =========================================================
# DISTRIBUTION PRINTING
# =========================================================
def print_distribution(submission, pred_cols, model_preds=None, naive_preds=None):
    print("\n=== Prediction distribution: final submission ===")
    print(submission[pred_cols].describe())

    print("\n=== Mean by week ===")
    print(submission[pred_cols].mean())

    print("\n=== Std by week ===")
    print(submission[pred_cols].std())

    print("\n=== Median by week ===")
    print(submission[pred_cols].median())

    print("\n=== Min by week ===")
    print(submission[pred_cols].min())

    print("\n=== Max by week ===")
    print(submission[pred_cols].max())

    print("\n=== Percentage of low predictions ===")
    for c in pred_cols:
        pct_eq_0 = (submission[c] <= 1e-8).mean()
        pct_lt_001 = (submission[c] < 0.01).mean()
        pct_lt_01 = (submission[c] < 0.1).mean()
        pct_lt_05 = (submission[c] < 0.5).mean()

        print(
            f"{c}: "
            f"<=1e-8={pct_eq_0:.3f}, "
            f"<0.01={pct_lt_001:.3f}, "
            f"<0.1={pct_lt_01:.3f}, "
            f"<0.5={pct_lt_05:.3f}"
        )

    print("\n=== Percentage of high predictions ===")
    for c in pred_cols:
        pct_ge_1 = (submission[c] >= 1).mean()
        pct_ge_2 = (submission[c] >= 2).mean()
        pct_ge_3 = (submission[c] >= 3).mean()
        pct_ge_4 = (submission[c] >= 4).mean()

        print(
            f"{c}: "
            f">=1={pct_ge_1:.3f}, "
            f">=2={pct_ge_2:.3f}, "
            f">=3={pct_ge_3:.3f}, "
            f">=4={pct_ge_4:.3f}"
        )

    if model_preds is not None:
        print("\n=== Model prediction distribution before ensemble ===")
        model_pred_df = pd.DataFrame(
            model_preds,
            columns=[f"model_pred_week{i}" for i in range(1, PRED_LEN + 1)]
        )
        print(model_pred_df.describe())

    if naive_preds is not None:
        print("\n=== Naive prediction distribution before ensemble ===")
        naive_pred_df = pd.DataFrame(
            naive_preds,
            columns=[f"naive_pred_week{i}" for i in range(1, PRED_LEN + 1)]
        )
        print(naive_pred_df.describe())


# =========================================================
# MAIN
# =========================================================
def main():
    print("Device:", DEVICE)
    print("PATCH_LEN:", PATCH_LEN)
    print("STRIDE:", STRIDE)
    print("DROPOUT:", DROPOUT)
    print("RECENT_RATIO:", RECENT_RATIO)
    print("ENSEMBLE_MODEL_WEIGHT:", ENSEMBLE_MODEL_WEIGHT)
    print("ENSEMBLE_NAIVE_WEIGHT:", ENSEMBLE_NAIVE_WEIGHT)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_df["region_id"] = train_df["region_id"].astype(str)
    test_df["region_id"] = test_df["region_id"].astype(str)

    train_df["date"] = train_df["date"].astype(str)
    test_df["date"] = test_df["date"].astype(str)

    if os.path.exists(SAMPLE_SUBMISSION):
        sample_df = pd.read_csv(SAMPLE_SUBMISSION)
        sample_df["region_id"] = sample_df["region_id"].astype(str)
    else:
        sample_df = None

    # -----------------------------------------------------
    # Fill base meteorological features first
    # -----------------------------------------------------
    for c in BASE_METEO_COLS:
        if c in train_df.columns:
            med = train_df[c].median()
            train_df[c] = train_df[c].fillna(med)
            test_df[c] = test_df[c].fillna(med)

    # -----------------------------------------------------
    # Add drought cumulative features
    # -----------------------------------------------------
    print("Adding drought cumulative features...")
    train_df = add_drought_features(train_df, BASE_METEO_COLS)
    test_df = add_drought_features(test_df, BASE_METEO_COLS)

    # -----------------------------------------------------
    # Feature columns after adding drought features
    # -----------------------------------------------------
    drop_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    # Ensure same columns in test
    missing_in_test = [c for c in feat_cols if c not in test_df.columns]
    if len(missing_in_test) > 0:
        raise ValueError(f"These feature columns are missing in test_df: {missing_in_test}")

    print("Number of feature columns:", len(feat_cols))
    print("Feature columns:")
    print(feat_cols)

    # Fill engineered feature NaN using train median
    for c in feat_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    all_regions = sorted(
        list(set(train_df["region_id"].unique()) | set(test_df["region_id"].unique()))
    )
    region_map = {r: i for i, r in enumerate(all_regions)}

    print("Number of regions:", len(region_map))

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    region_data = build_region_data(train_df, feat_cols, region_map)

    train_samples, val_samples = build_train_val_samples(region_data)

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    train_dataset = DisasterWindowDataset(region_data, train_samples)
    val_dataset = DisasterWindowDataset(region_data, val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = PatchTST(
        n_feat=len(feat_cols),
        n_region=len(region_map)
    ).to(DEVICE)

    print(model)

    loss_fn = nn.SmoothL1Loss(beta=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_mae = 1e9

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_mae, val_rmse, week_mae = evaluate(model, val_loader, loss_fn)

        scheduler.step()

        week_str = " | ".join([
            f"W{i+1}: {m:.5f}" for i, m in enumerate(week_mae)
        ])

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MAE: {val_mae:.5f} | "
            f"Val RMSE: {val_rmse:.5f} | "
            f"{week_str}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH} | Best MAE: {best_mae:.5f}")

    # =====================================================
    # Load best model
    # =====================================================
    print("\nLoading best model...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    # =====================================================
    # Test inference
    # =====================================================
    X_test, R_test, region_names = build_test_samples(
        test_df,
        feat_cols,
        region_map,
        sample_df
    )

    print("Test X:", X_test.shape)

    test_dataset = DisasterTestDataset(X_test, R_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model_preds = inference(model, test_loader)

    print("Model pred shape:", model_preds.shape)

    # =====================================================
    # Naive baseline
    # =====================================================
    naive_preds = build_naive_pred(train_df, region_names)

    print("Naive pred shape:", naive_preds.shape)

    np.save("model_preds_drought_features.npy", model_preds)
    np.save("naive_preds_drought_features.npy", naive_preds)
    pd.DataFrame({"region_id": region_names}).to_csv(
        "region_names_drought_features.csv",
        index=False
    )

    print("Saved model_preds_drought_features.npy")
    print("Saved naive_preds_drought_features.npy")
    print("Saved region_names_drought_features.csv")

    # =====================================================
    # Ensemble
    # =====================================================
    final_preds = (
        ENSEMBLE_MODEL_WEIGHT * model_preds
        + ENSEMBLE_NAIVE_WEIGHT * naive_preds
    )

    final_preds = np.clip(final_preds, 0, 5)

    # =====================================================
    # Submission
    # =====================================================
    if sample_df is not None:
        submission = sample_df.copy()
        pred_cols = [c for c in submission.columns if c != "region_id"]

        if len(pred_cols) != PRED_LEN:
            print("Warning: sample_submission columns do not match PRED_LEN")
            print("Columns:", submission.columns.tolist())

        for i, col in enumerate(pred_cols[:PRED_LEN]):
            submission[col] = final_preds[:, i]

    else:
        pred_cols = [f"pred_week{i}" for i in range(1, PRED_LEN + 1)]
        submission = pd.DataFrame({
            "region_id": region_names,
            "pred_week1": final_preds[:, 0],
            "pred_week2": final_preds[:, 1],
            "pred_week3": final_preds[:, 2],
            "pred_week4": final_preds[:, 3],
            "pred_week5": final_preds[:, 4],
        })

    submission.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    print_distribution(
        submission=submission,
        pred_cols=pred_cols,
        model_preds=model_preds,
        naive_preds=naive_preds
    )

    print("\n=== Ensemble setting ===")
    print(f"ENSEMBLE_MODEL_WEIGHT = {ENSEMBLE_MODEL_WEIGHT}")
    print(f"ENSEMBLE_NAIVE_WEIGHT = {ENSEMBLE_NAIVE_WEIGHT}")
    print(f"Best validation MAE = {best_mae:.5f}")


if __name__ == "__main__":
    main()