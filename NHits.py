# =========================================================
# N-BEATS / N-HiTS Style MLP
# Natural Disaster Severity Prediction
#
# Version:
#   - Raw weather features
#   - Stable region score prior:
#       region_score_mean
#       region_high_ratio
#       region_zero_ratio
#   - N-HiTS-like residual MLP blocks
#   - Fixed ensemble:
#       0.70 * model_pred + 0.30 * naive_latest5
#   - No Markov
#   - No kNN
#   - No Ridge calibration
#   - No residual lookup
#   - No region weather prior
#   - No region score std/max/recent stats
# =========================================================

import os
import gc
import random
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_DIR = "./nhits_mlp_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(OUT_DIR, "best_nhits_mlp.pth")

ID_COL = "region_id"
DATE_COL = "date"
TARGET_COL = "score"

PRED_COLS = [
    "pred_week1",
    "pred_week2",
    "pred_week3",
    "pred_week4",
    "pred_week5",
]

BASE_FEATURE_COLS = [
    "wind", "wind_min", "wind_max", "wind_range",
    "humidity",
    "tmp", "tmp_range", "tmp_max", "tmp_min",
    "surf_tmp", "surf_pre",
    "dp_tmp", "wb_tmp",
    "prec",
]

REGION_STAT_COLS = [
    "region_score_mean",
    "region_high_ratio",
    "region_zero_ratio",
]

INPUT_LEN = 91
PRED_LEN = 5

SEED = 42

RECENT_RATIO = 0.5
VAL_RATIO = 0.15

FINAL_TRAIN = False

BATCH_SIZE = 128
EPOCHS = 35
PATIENCE = 6
MIN_DELTA = 1e-4

LR = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REGION_EMB_DIM = 64

# N-HiTS / N-BEATS-like MLP config
NHITS_HIDDEN_DIM = 512
NHITS_DROPOUT = 0.15
NHITS_BLOCKS = 4

# pooling scales for multi-resolution blocks
# 1 = original resolution
# 3/7/14 = coarser temporal summaries
NHITS_POOL_SIZES = [1, 3, 7, 14]

# final fusion MLP
FUSION_HIDDEN_DIM = 256

USE_AUGMENT = True
NOISE_STD = 0.01

TIME_MASK_PROB = 0.10
TIME_MASK_MIN_LEN = 3
TIME_MASK_MAX_LEN = 7

FEATURE_MASK_PROB = 0.05
FEATURE_MASK_MAX_NUM = 2

ENSEMBLE_MODEL_WEIGHT = 0.70
ENSEMBLE_NAIVE_WEIGHT = 0.30


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# =========================================================
# REGION SCORE PRIOR FEATURES
# =========================================================
def add_region_score_stats(train_df, test_df):
    """
    Add stable region-level historical score prior features.

    Added:
      - region_score_mean
      - region_high_ratio: ratio of score >= 2
      - region_zero_ratio: ratio of score == 0
    """

    tmp = train_df[[ID_COL, TARGET_COL]].copy()
    tmp[ID_COL] = tmp[ID_COL].astype(str)
    tmp[TARGET_COL] = pd.to_numeric(tmp[TARGET_COL], errors="coerce")

    stats = (
        tmp.groupby(ID_COL)
        .agg(
            region_score_mean=(TARGET_COL, "mean"),
        )
        .reset_index()
    )

    high_ratio = (
        tmp.assign(score_high=(tmp[TARGET_COL] >= 2).astype(float))
        .groupby(ID_COL)["score_high"]
        .mean()
        .reset_index()
        .rename(columns={"score_high": "region_high_ratio"})
    )

    zero_ratio = (
        tmp.assign(score_zero=(tmp[TARGET_COL] == 0).astype(float))
        .groupby(ID_COL)["score_zero"]
        .mean()
        .reset_index()
        .rename(columns={"score_zero": "region_zero_ratio"})
    )

    stats = stats.merge(high_ratio, on=ID_COL, how="left")
    stats = stats.merge(zero_ratio, on=ID_COL, how="left")

    train_df = train_df.merge(stats, on=ID_COL, how="left")
    test_df = test_df.merge(stats, on=ID_COL, how="left")

    for c in REGION_STAT_COLS:
        med = train_df[c].median()
        if pd.isna(med):
            med = 0.0

        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    return train_df, test_df


# =========================================================
# DATASET
# =========================================================
class DisasterWindowDataset(Dataset):
    def __init__(self, region_data, samples, augment=False):
        self.region_data = region_data
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment_x(self, x):
        x = x.copy()

        if NOISE_STD > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=NOISE_STD,
                size=x.shape,
            ).astype(np.float32)
            x = x + noise

        if np.random.rand() < TIME_MASK_PROB:
            mask_len = np.random.randint(
                TIME_MASK_MIN_LEN,
                TIME_MASK_MAX_LEN + 1,
            )
            mask_len = min(mask_len, x.shape[0])

            start = np.random.randint(0, x.shape[0] - mask_len + 1)
            x[start:start + mask_len, :] = 0.0

        if np.random.rand() < FEATURE_MASK_PROB:
            c = x.shape[1]
            n_mask = np.random.randint(
                1,
                min(FEATURE_MASK_MAX_NUM, c) + 1,
            )
            cols = np.random.choice(c, size=n_mask, replace=False)
            x[:, cols] = 0.0

        return x.astype(np.float32)

    def __getitem__(self, idx):
        rid, start, end, target_idx = self.samples[idx]

        feat = self.region_data[rid]["feat"]
        score = self.region_data[rid]["score"]

        x = feat[start:end].astype(np.float32)
        y = score[target_idx].astype(np.float32)

        if self.augment:
            x = self._augment_x(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(rid, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
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
# DATA BUILDING
# =========================================================
def build_region_data(train_df, feat_cols, region_map):
    region_data = {}

    train_df = train_df.sort_values([ID_COL, DATE_COL])

    for region, g in train_df.groupby(ID_COL):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        rid = region_map[region]

        region_data[rid] = {
            "region_name": region,
            "feat": g[feat_cols].values.astype(np.float32),
            "score": g[TARGET_COL].values.astype(np.float32),
        }

    return region_data


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

        use_start = int(len(all_samples) * (1.0 - RECENT_RATIO))
        recent_samples = all_samples[use_start:]

        split = int(len(recent_samples) * (1.0 - VAL_RATIO))

        train_samples.extend(recent_samples[:split])
        val_samples.extend(recent_samples[split:])

    return train_samples, val_samples


def build_test_samples(test_df, feat_cols, region_map, sample_df=None):
    test_df = test_df.sort_values([ID_COL, DATE_COL])

    X_list = []
    R_list = []
    region_names = []

    if sample_df is not None and ID_COL in sample_df.columns:
        order_regions = sample_df[ID_COL].astype(str).tolist()
    else:
        order_regions = sorted(test_df[ID_COL].astype(str).unique())

    grouped = {
        str(r): g.sort_values(DATE_COL).reset_index(drop=True)
        for r, g in test_df.groupby(ID_COL)
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

    return (
        np.array(X_list, dtype=np.float32),
        np.array(R_list, dtype=np.int64),
        region_names,
    )


# =========================================================
# NAIVE BASELINE
# =========================================================
def build_test_naive_pred(train_df, region_names):
    """
    Latest-5 score baseline for fixed ensemble.
    """

    naive_preds = []

    df = train_df.copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df = df.sort_values([ID_COL, DATE_COL])

    for region in region_names:
        g = df[df[ID_COL] == str(region)].sort_values(DATE_COL)
        scores = g[TARGET_COL].dropna().values.astype(np.float32)

        if len(scores) >= PRED_LEN:
            pred = scores[-PRED_LEN:]
        elif len(scores) > 0:
            pred = np.full(PRED_LEN, scores[-1], dtype=np.float32)
        else:
            pred = np.zeros(PRED_LEN, dtype=np.float32)

        naive_preds.append(pred)

    naive_preds = np.array(naive_preds, dtype=np.float32)
    naive_preds = np.clip(naive_preds, 0, 5)

    return naive_preds


def build_val_naive_pred(region_data, samples):
    """
    For validation samples, use the 5 known scores before target start.
    """

    naive_preds = []

    for rid, start, end, target_idx in samples:
        score = region_data[rid]["score"]

        hist_start = max(0, end - PRED_LEN)
        hist = score[hist_start:end]
        hist = hist[~np.isnan(hist)]

        if len(hist) >= PRED_LEN:
            pred = hist[-PRED_LEN:]
        elif len(hist) > 0:
            pred = np.full(PRED_LEN, hist[-1], dtype=np.float32)
        else:
            pred = np.zeros(PRED_LEN, dtype=np.float32)

        naive_preds.append(pred.astype(np.float32))

    naive_preds = np.array(naive_preds, dtype=np.float32)
    naive_preds = np.clip(naive_preds, 0, 5)

    return naive_preds


# =========================================================
# N-BEATS / N-HiTS STYLE MODEL
# =========================================================
class NHiTSBlock(nn.Module):
    """
    N-HiTS-like block:
      - optional temporal pooling
      - MLP
      - backcast output for residual update
      - forecast output for horizon prediction
    """

    def __init__(
        self,
        input_len,
        n_feat,
        pred_len,
        region_emb_dim,
        pool_size=1,
        hidden_dim=512,
        dropout=0.15,
    ):
        super().__init__()

        self.input_len = input_len
        self.n_feat = n_feat
        self.pred_len = pred_len
        self.pool_size = pool_size

        if pool_size <= 1:
            self.pooled_len = input_len
        else:
            self.pooled_len = math.ceil(input_len / pool_size)

        pooled_dim = self.pooled_len * n_feat
        mlp_input_dim = pooled_dim + region_emb_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.backcast_head = nn.Linear(hidden_dim // 2, input_len * n_feat)
        self.forecast_head = nn.Linear(hidden_dim // 2, pred_len)

    def forward(self, x, region_emb):
        """
        x: [B, T, C]
        region_emb: [B, E]
        """

        b, t, c = x.shape

        if self.pool_size <= 1:
            pooled = x
        else:
            # [B, T, C] -> [B, C, T]
            z = x.permute(0, 2, 1)

            pooled = F.avg_pool1d(
                z,
                kernel_size=self.pool_size,
                stride=self.pool_size,
                ceil_mode=True,
            )

            # [B, C, T_pooled] -> [B, T_pooled, C]
            pooled = pooled.permute(0, 2, 1)

        flat = pooled.reshape(b, -1)
        feat = torch.cat([flat, region_emb], dim=1)

        h = self.mlp(feat)

        backcast = self.backcast_head(h)
        backcast = backcast.reshape(b, self.input_len, self.n_feat)

        forecast = self.forecast_head(h)

        backcast = torch.nan_to_num(backcast, nan=0.0, posinf=0.0, neginf=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0, posinf=5.0, neginf=0.0)

        return backcast, forecast


class NHiTSMLPModel(nn.Module):
    def __init__(self, n_feat, n_region):
        super().__init__()

        self.region_emb = nn.Embedding(n_region, REGION_EMB_DIM)

        pool_sizes = NHITS_POOL_SIZES[:NHITS_BLOCKS]

        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_len=INPUT_LEN,
                n_feat=n_feat,
                pred_len=PRED_LEN,
                region_emb_dim=REGION_EMB_DIM,
                pool_size=pool_size,
                hidden_dim=NHITS_HIDDEN_DIM,
                dropout=NHITS_DROPOUT,
            )
            for pool_size in pool_sizes
        ])

        fusion_dim = PRED_LEN * len(pool_sizes) + PRED_LEN + REGION_EMB_DIM

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, FUSION_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(NHITS_DROPOUT),

            nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(NHITS_DROPOUT),

            nn.Linear(FUSION_HIDDEN_DIM // 2, PRED_LEN),
        )

    def forward(self, x, r):
        """
        x: [B, 91, C]
        r: [B]
        """

        region_feat = self.region_emb(r)

        residual = x
        forecasts = []

        for block in self.blocks:
            backcast, forecast = block(residual, region_feat)

            residual = residual - backcast
            forecasts.append(forecast)

        stacked_forecast = torch.stack(forecasts, dim=1)   # [B, blocks, 5]
        sum_forecast = stacked_forecast.sum(dim=1)         # [B, 5]

        fusion = torch.cat(
            [
                stacked_forecast.reshape(x.size(0), -1),
                sum_forecast,
                region_feat,
            ],
            dim=1,
        )

        correction = self.fusion_head(fusion)

        # Stable final prediction:
        # use sum forecast as base, fusion head as correction
        pred = sum_forecast + 0.5 * correction

        pred = torch.nan_to_num(pred, nan=0.0, posinf=5.0, neginf=0.0)

        return pred


# =========================================================
# TRAIN / EVAL / INFERENCE
# =========================================================
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()

    total_loss = 0.0
    total_seen = 0

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

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

    return total_loss / max(1, total_seen)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_seen = 0

    preds = []
    trues = []

    for x, r, y in loader:
        x = x.to(DEVICE)
        r = r.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x, r)
        loss = loss_fn(pred, y)

        pred = torch.clamp(pred, 0, 5)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    week_mae = np.mean(np.abs(preds - trues), axis=0)

    return total_loss / max(1, total_seen), mae, rmse, week_mae, preds, trues


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
# SAVE / UTILS
# =========================================================
def print_prediction_stats(name, pred):
    pred = np.asarray(pred)

    print(f"\n========== {name} Stats ==========")
    print("shape:", pred.shape)
    print("min  :", float(np.min(pred)))
    print("max  :", float(np.max(pred)))
    print("mean :", float(np.mean(pred)))
    print("std  :", float(np.std(pred)))
    print("near-zero ratio (<0.05):", float(np.mean(pred < 0.05)))
    print("zero ratio (=0):", float(np.mean(pred == 0.0)))
    print("week mean:", np.round(pred.mean(axis=0), 5))
    print("week std :", np.round(pred.std(axis=0), 5))


def print_score_distribution(train_df):
    scores = train_df[TARGET_COL].dropna().values.astype(np.float32)

    print("\n========== Train Score Distribution ==========")
    print("count:", len(scores))
    print("min  :", float(np.min(scores)))
    print("max  :", float(np.max(scores)))
    print("mean :", float(np.mean(scores)))
    print("std  :", float(np.std(scores)))
    print("zero ratio:", float(np.mean(scores == 0)))
    print("<0.05 ratio:", float(np.mean(scores < 0.05)))
    print(">1 ratio:", float(np.mean(scores > 1)))
    print(">2 ratio:", float(np.mean(scores > 2)))
    print(">3 ratio:", float(np.mean(scores > 3)))
    print(">4 ratio:", float(np.mean(scores > 4)))


def save_submission(sample_df, region_names, preds, out_path):
    preds = np.nan_to_num(preds, nan=0.0, posinf=5.0, neginf=0.0)
    preds = np.clip(preds, 0, 5)

    if sample_df is not None:
        submission = sample_df.copy()
        submission[ID_COL] = submission[ID_COL].astype(str)

        pred_cols = [c for c in submission.columns if c != ID_COL]

        if len(pred_cols) != PRED_LEN:
            pred_cols = PRED_COLS
            submission = submission[[ID_COL]].copy()

        for i, col in enumerate(pred_cols[:PRED_LEN]):
            submission[col] = preds[:, i]

    else:
        submission = pd.DataFrame({
            ID_COL: region_names,
            PRED_COLS[0]: preds[:, 0],
            PRED_COLS[1]: preds[:, 1],
            PRED_COLS[2]: preds[:, 2],
            PRED_COLS[3]: preds[:, 3],
            PRED_COLS[4]: preds[:, 4],
        })

    submission.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Range:", float(preds.min()), float(preds.max()))
    print(submission.head())
    print()


def save_all_submissions(
    sample_df,
    region_names,
    train_df,
    model_preds,
    naive_preds,
):
    """
    Save raw model and fixed 0.70/0.30 ensemble.
    """

    # Raw N-HiTS style model
    save_submission(
        sample_df=sample_df,
        region_names=region_names,
        preds=model_preds,
        out_path=os.path.join(OUT_DIR, "submission_nhits_mlp_raw.csv"),
    )

    # Fixed 0.70 / 0.30 ensemble
    fixed_pred = (
        ENSEMBLE_MODEL_WEIGHT * model_preds
        + ENSEMBLE_NAIVE_WEIGHT * naive_preds
    )
    fixed_pred = np.clip(fixed_pred, 0, 5)

    save_submission(
        sample_df=sample_df,
        region_names=region_names,
        preds=fixed_pred,
        out_path=os.path.join(
            OUT_DIR,
            f"submission_nhits_mlp_model{ENSEMBLE_MODEL_WEIGHT:.2f}_naive{ENSEMBLE_NAIVE_WEIGHT:.2f}.csv",
        ),
    )

    # qclip on fixed ensemble
    scores = train_df[TARGET_COL].dropna().values.astype(np.float32)

    for q_low, q_high in [(0.001, 0.999), (0.005, 0.995)]:
        lo = float(np.quantile(scores, q_low))
        hi = float(np.quantile(scores, q_high))

        fixed_clip = np.clip(fixed_pred, lo, hi)
        fixed_clip = np.clip(fixed_clip, 0, 5)

        save_submission(
            sample_df=sample_df,
            region_names=region_names,
            preds=fixed_clip,
            out_path=os.path.join(
                OUT_DIR,
                f"submission_nhits_mlp_model070_naive030_qclip_{q_low:.3f}_{q_high:.3f}.csv",
            ),
        )

    # default submission.csv
    save_submission(
        sample_df=sample_df,
        region_names=region_names,
        preds=fixed_pred,
        out_path="submission.csv",
    )

    print("\nRecommended files:")
    print(os.path.join(OUT_DIR, f"submission_nhits_mlp_model{ENSEMBLE_MODEL_WEIGHT:.2f}_naive{ENSEMBLE_NAIVE_WEIGHT:.2f}.csv"))
    print(os.path.join(OUT_DIR, "submission_nhits_mlp_raw.csv"))
    print(os.path.join(OUT_DIR, "submission_nhits_mlp_model070_naive030_qclip_0.001_0.999.csv"))
    print("Default saved as: submission.csv")


# =========================================================
# MAIN
# =========================================================
def main():
    print("Device:", DEVICE)
    print("N-BEATS / N-HiTS Style MLP")
    print("SEED:", SEED)
    print("RECENT_RATIO:", RECENT_RATIO)
    print("VAL_RATIO:", VAL_RATIO)
    print("FINAL_TRAIN:", FINAL_TRAIN)
    print("USE_AUGMENT:", USE_AUGMENT)
    print("NHITS_HIDDEN_DIM:", NHITS_HIDDEN_DIM)
    print("NHITS_BLOCKS:", NHITS_BLOCKS)
    print("NHITS_POOL_SIZES:", NHITS_POOL_SIZES)
    print("ENSEMBLE_MODEL_WEIGHT:", ENSEMBLE_MODEL_WEIGHT)
    print("ENSEMBLE_NAIVE_WEIGHT:", ENSEMBLE_NAIVE_WEIGHT)

    # -------------------------
    # Load data
    # -------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_df[ID_COL] = train_df[ID_COL].astype(str)
    test_df[ID_COL] = test_df[ID_COL].astype(str)

    train_df[DATE_COL] = train_df[DATE_COL].astype(str)
    test_df[DATE_COL] = test_df[DATE_COL].astype(str)

    train_df[TARGET_COL] = pd.to_numeric(train_df[TARGET_COL], errors="coerce")

    if os.path.exists(SAMPLE_SUBMISSION):
        sample_df = pd.read_csv(SAMPLE_SUBMISSION)
        sample_df[ID_COL] = sample_df[ID_COL].astype(str)
    else:
        sample_df = None

    print("\n========== Date Range Check ==========")
    print("Train date range:")
    print(train_df.groupby(ID_COL)[DATE_COL].agg(["min", "max"]).head())
    print("Test date range:")
    print(test_df.groupby(ID_COL)[DATE_COL].agg(["min", "max"]).head())

    print_score_distribution(train_df)

    # -------------------------
    # Base weather preprocessing
    # -------------------------
    for c in BASE_FEATURE_COLS:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

        med = train_df[c].median()
        if pd.isna(med):
            med = 0.0

        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    # -------------------------
    # Add stable region score prior
    # -------------------------
    train_df, test_df = add_region_score_stats(train_df, test_df)

    print("\n========== Region Score Stats Preview ==========")
    preview_cols = [ID_COL] + REGION_STAT_COLS
    print(train_df[preview_cols].drop_duplicates(ID_COL).head())

    # -------------------------
    # Feature preprocessing
    # -------------------------
    feat_cols = BASE_FEATURE_COLS + REGION_STAT_COLS

    for c in feat_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

        med = train_df[c].median()
        if pd.isna(med):
            med = 0.0

        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    all_regions = sorted(
        list(set(train_df[ID_COL].unique()) | set(test_df[ID_COL].unique()))
    )

    region_map = {r: i for i, r in enumerate(all_regions)}

    print("\n========== Data Info ==========")
    print("Number of regions:", len(region_map))
    print("Base features:", len(BASE_FEATURE_COLS))
    print("Region stat features:", len(REGION_STAT_COLS))
    print("Total features:", len(feat_cols))
    print("Feature columns:", feat_cols)

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    train_df[feat_cols] = train_df[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
    test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # -------------------------
    # Build samples
    # -------------------------
    region_data = build_region_data(train_df, feat_cols, region_map)
    train_samples, val_samples = build_train_val_samples(region_data)

    print("\n========== Samples ==========")
    print("Original train samples:", len(train_samples))
    print("Original val samples:", len(val_samples))

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("No train/val samples. Check RECENT_RATIO / VAL_RATIO.")

    if FINAL_TRAIN:
        print("FINAL_TRAIN=True: using train + val for training.")
        train_samples = train_samples + val_samples

    print("Used train samples:", len(train_samples))
    print("Used val samples:", len(val_samples))

    train_dataset = DisasterWindowDataset(
        region_data=region_data,
        samples=train_samples,
        augment=USE_AUGMENT,
    )

    val_dataset = DisasterWindowDataset(
        region_data=region_data,
        samples=val_samples,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = NHiTSMLPModel(
        n_feat=len(feat_cols),
        n_region=len(region_map),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Model Params ==========")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    loss_fn = nn.SmoothL1Loss(beta=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_mae = 1e9
    bad_epochs = 0

    # -------------------------
    # Training
    # -------------------------
    print("\n========== Training ==========")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_mae, val_rmse, week_mae, _, _ = evaluate(
            model,
            val_loader,
            loss_fn,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR={optimizer.param_groups[0]['lr']:.8f} | "
            f"TrainLoss={train_loss:.5f} | "
            f"ValLoss={val_loss:.5f} | "
            f"ValMAE={val_mae:.5f} | "
            f"ValRMSE={val_rmse:.5f} | "
            f"WeekMAE={np.round(week_mae, 5)}"
        )

        if FINAL_TRAIN:
            torch.save(model.state_dict(), SAVE_PATH)
            continue

        if val_mae < best_mae - MIN_DELTA:
            best_mae = val_mae
            bad_epochs = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH} | Best Val MAE={best_mae:.5f}")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    # -------------------------
    # Load best
    # -------------------------
    print("\nLoading best model...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    # -------------------------
    # Validation check
    # -------------------------
    (
        val_loss,
        val_mae,
        val_rmse,
        val_week_mae,
        val_model_preds,
        val_trues,
    ) = evaluate(model, val_loader, loss_fn)

    val_naive_preds = build_val_naive_pred(
        region_data=region_data,
        samples=val_samples,
    )

    val_fixed_preds = (
        ENSEMBLE_MODEL_WEIGHT * val_model_preds
        + ENSEMBLE_NAIVE_WEIGHT * val_naive_preds
    )
    val_fixed_preds = np.clip(val_fixed_preds, 0, 5)

    print("\n========== Validation Fixed Ensemble Check ==========")
    print("Raw N-HiTS MLP Val MAE:", float(np.mean(np.abs(val_model_preds - val_trues))))
    print("Naive Val MAE:", float(np.mean(np.abs(val_naive_preds - val_trues))))
    print("Fixed model0.70 naive0.30 Val MAE:", float(np.mean(np.abs(val_fixed_preds - val_trues))))

    # -------------------------
    # Test inference
    # -------------------------
    X_test, R_test, region_names = build_test_samples(
        test_df=test_df,
        feat_cols=feat_cols,
        region_map=region_map,
        sample_df=sample_df,
    )

    print("\n========== Test ==========")
    print("Test X:", X_test.shape)
    print("Test R:", R_test.shape)
    print("Regions:", len(region_names))

    test_dataset = DisasterTestDataset(
        X=X_test,
        R=R_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    test_model_preds = inference(model, test_loader)
    test_naive_preds = build_test_naive_pred(train_df, region_names)

    np.save(os.path.join(OUT_DIR, "test_nhits_mlp_model_preds.npy"), test_model_preds)
    np.save(os.path.join(OUT_DIR, "test_naive_preds.npy"), test_naive_preds)

    print_prediction_stats("N-HiTS MLP model prediction", test_model_preds)
    print_prediction_stats("Naive latest-5 prediction", test_naive_preds)

    # -------------------------
    # Save submissions
    # -------------------------
    save_all_submissions(
        sample_df=sample_df,
        region_names=region_names,
        train_df=train_df,
        model_preds=test_model_preds,
        naive_preds=test_naive_preds,
    )

    gc.collect()


if __name__ == "__main__":
    main()