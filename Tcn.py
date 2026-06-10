# =========================================================
# TCN Regression + Drought Cumulative Features + Naive Ensemble
# Natural Disaster Severity Prediction
#
# Output folder version + Early Stopping
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

OUTPUT_DIR = "./outputs_tcn_drought_k7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(OUTPUT_DIR, "best_tcn_drought_features.pth")
OUT_CSV = os.path.join(OUTPUT_DIR, "TCN_DroughtFeatures_70_30.csv")

MODEL_PREDS_NPY = os.path.join(OUTPUT_DIR, "model_preds_tcn_drought_features.npy")
NAIVE_PREDS_NPY = os.path.join(OUTPUT_DIR, "naive_preds_tcn_drought_features.npy")
REGION_NAMES_CSV = os.path.join(OUTPUT_DIR, "region_names_tcn_drought_features.csv")

INPUT_LEN = 91
PRED_LEN = 5

# TCN setting
TCN_CHANNELS = [128, 128, 256, 256]
TCN_KERNEL_SIZE = 7
DROPOUT = 0.1
HIDDEN_DIM = 256

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.15
RECENT_RATIO = 0.5

ENSEMBLE_MODEL_WEIGHT = 0.7
ENSEMBLE_NAIVE_WEIGHT = 0.3

EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-5

SEED = 42
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


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
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    if "prec" in df.columns:
        df["prec_log"] = np.log1p(
            np.clip(df["prec"].astype(np.float32), a_min=0, a_max=None)
        )

    if "tmp" in df.columns and "humidity" in df.columns:
        humidity_stress = np.clip(
            100.0 - df["humidity"].astype(np.float32),
            a_min=0,
            a_max=None
        )
        df["hot_dry_index"] = df["tmp"].astype(np.float32) * humidity_stress

    if "tmp_max" in df.columns and "humidity" in df.columns:
        humidity_stress = np.clip(
            100.0 - df["humidity"].astype(np.float32),
            a_min=0,
            a_max=None
        )
        df["hot_dry_max_index"] = df["tmp_max"].astype(np.float32) * humidity_stress

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

    if "prec_roll7_sum" in df.columns and "prec_roll28_sum" in df.columns:
        df["prec_roll7_minus_roll28_avg"] = (
            df["prec_roll7_sum"] - df["prec_roll28_sum"] / 4.0
        )

    if "prec_roll14_sum" in df.columns and "prec_roll56_sum" in df.columns:
        df["prec_roll14_minus_roll56_avg"] = (
            df["prec_roll14_sum"] - df["prec_roll56_sum"] / 4.0
        )

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
# TCN MODEL
# =========================================================
class Chomp1d(nn.Module):
    """
    Remove extra padding on the right side to keep causal convolution length.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.final_act = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.final_act(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, channels, kernel_size=3, dropout=0.1):
        super().__init__()

        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else channels[i - 1]
            out_ch = channels[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNRegressor(nn.Module):
    """
    TCN for multi-step score regression.

    Input:
        x: [B, 91, C]
        r: [B]

    Output:
        pred: [B, 5]
    """
    def __init__(self, n_feat, n_region):
        super().__init__()

        self.tcn = TemporalConvNet(
            input_channels=n_feat,
            channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=DROPOUT
        )

        last_ch = TCN_CHANNELS[-1]

        self.region_emb = nn.Embedding(n_region, HIDDEN_DIM)

        self.head = nn.Sequential(
            nn.LayerNorm(last_ch * 2 + HIDDEN_DIM),
            nn.Linear(last_ch * 2 + HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, PRED_LEN)
        )

    def forward(self, x, r):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        feat = self.tcn(x)  # [B, hidden, T]

        # last timestep + global average pooling
        last_feat = feat[:, :, -1]
        avg_feat = feat.mean(dim=2)

        r_emb = self.region_emb(r)

        out = torch.cat([last_feat, avg_feat, r_emb], dim=1)

        return self.head(out)


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
    print("Model: TCN")
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("INPUT_LEN:", INPUT_LEN)
    print("PRED_LEN:", PRED_LEN)
    print("TCN_CHANNELS:", TCN_CHANNELS)
    print("TCN_KERNEL_SIZE:", TCN_KERNEL_SIZE)
    print("DROPOUT:", DROPOUT)
    print("RECENT_RATIO:", RECENT_RATIO)
    print("ENSEMBLE_MODEL_WEIGHT:", ENSEMBLE_MODEL_WEIGHT)
    print("ENSEMBLE_NAIVE_WEIGHT:", ENSEMBLE_NAIVE_WEIGHT)
    print("EARLY_STOPPING_PATIENCE:", EARLY_STOPPING_PATIENCE)
    print("MIN_DELTA:", MIN_DELTA)

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

    for c in BASE_METEO_COLS:
        if c in train_df.columns:
            med = train_df[c].median()
            train_df[c] = train_df[c].fillna(med)
            test_df[c] = test_df[c].fillna(med)

    print("Adding drought cumulative features...")
    train_df = add_drought_features(train_df, BASE_METEO_COLS)
    test_df = add_drought_features(test_df, BASE_METEO_COLS)

    drop_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    missing_in_test = [c for c in feat_cols if c not in test_df.columns]
    if len(missing_in_test) > 0:
        raise ValueError(f"These feature columns are missing in test_df: {missing_in_test}")

    print("Number of feature columns:", len(feat_cols))
    print("Feature columns:")
    print(feat_cols)

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

    model = TCNRegressor(
        n_feat=len(feat_cols),
        n_region=len(region_map)
    ).to(DEVICE)

    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

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
    best_epoch = 0
    patience_counter = 0

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

        improved = val_mae < (best_mae - MIN_DELTA)

        if improved:
            best_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH} | Best MAE: {best_mae:.5f}")
        else:
            patience_counter += 1
            print(
                f"No improvement. "
                f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best epoch = {best_epoch}, Best Val MAE = {best_mae:.5f}"
            )
            break

    print("\nLoading best model...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

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

    naive_preds = build_naive_pred(train_df, region_names)
    print("Naive pred shape:", naive_preds.shape)

    np.save(MODEL_PREDS_NPY, model_preds)
    np.save(NAIVE_PREDS_NPY, naive_preds)

    pd.DataFrame({"region_id": region_names}).to_csv(
        REGION_NAMES_CSV,
        index=False
    )

    print(f"Saved {MODEL_PREDS_NPY}")
    print(f"Saved {NAIVE_PREDS_NPY}")
    print(f"Saved {REGION_NAMES_CSV}")

    final_preds = (
        ENSEMBLE_MODEL_WEIGHT * model_preds
        + ENSEMBLE_NAIVE_WEIGHT * naive_preds
    )

    final_preds = np.clip(final_preds, 0, 5)

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
    print(f"Best epoch = {best_epoch}")

    print("\n=== Output files ===")
    print(f"Best model: {SAVE_PATH}")
    print(f"Submission: {OUT_CSV}")
    print(f"Model preds: {MODEL_PREDS_NPY}")
    print(f"Naive preds: {NAIVE_PREDS_NPY}")
    print(f"Region names: {REGION_NAMES_CSV}")


if __name__ == "__main__":
    main()