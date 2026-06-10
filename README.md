## DM2026-Final-Project: Natural Disaster Severity Prediction

* **Name and Student ID:** 314551087 黃奕睿, 314554023 李思恩, 314554021 王品喻

## Abstract

This project aims to predict future drought severity scores for different geographic regions using historical meteorological data. We formulate the task as a multi-step time-series forecasting problem, where the past 91 days of daily weather observations are used to predict the severity scores for the next five prediction weeks. To improve prediction performance, we propose a multi-branch forecasting pipeline that combines different feature representations. One branch focuses on drought-related meteorological features, such as rolling precipitation statistics, hot-dry indices, and short-term versus long-term weather trends, while another branch incorporates region-level historical score priors to capture long-term regional characteristics. The proposed method further combines multiple forecasting models with a Naive historical reference and ensemble learning to improve prediction stability. Empirical post-processing is also applied to adjust the prediction distribution, clip outputs to the valid score range, and reduce unstable low-score predictions. Experimental results on the public leaderboard show that our method achieves a Public MAE of 0.7590 and ranks 2nd, demonstrating the effectiveness of combining diverse feature engineering strategies, ensemble learning, and post-processing for drought severity prediction.



## Environment Setup

### Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
.
├── Tcn.py               # Training baseline
├── PatchTS.py           # Training V2
├── NHits.py             # Training V3
├── per_week_weight.py   # Training V4 (best)
├── ensemble2sub.py      # inference baseline
├── infer_V2.py          # inference V2
├── requirements.txt     # Project dependencies
└── data/                # Dataset directory
```

## Usage

### Training

```bash
python train_V4.py 
```

### Configuration
```bash
# DATA PATH
DATA_ROOT = "./hw4_realse_dataset"
TRAIN_DEGRADED_DIR = os.path.join(DATA_ROOT, "train", "degraded")
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train", "clean")
```
Hyperparameter:
- `Image Size`: 256 × 256
- `Epochs`: 100
- `Batch Size`: 8
- `Learning Rate`: 1e-4
- `Weight Decay`: 1e-4
- `Validation Ratio`: 0.1
- `Optimizer`: AdamW
- `Scheduler`: CosineAnnealingLR
- `Loss Function`: Charbonnier Loss

### Inference

```bash
python infer_V4.py
```
## Strategy and Adjustments

The following modifications and strategies are applied in the model and training process:

1. Uses residual learning to predict only the degradation correction.
2. Uses Charbonnier Loss for stable image restoration training.
3. Uses	DetailRefineBlock.
4. Uses	FrequencyEnhanceBlock.
5. Uses	Gated skip fusion.

## Additional experiments

### V1~3 (select the version you want)
```bash
python train_V<1~3>.py # select the version you want

python infer_V<1~3>.py
```

## Performance

- Public test data PSNR : 30.90

| Model | Best Val PSNR | Scores |
|------|------|------|
| PromptIR | 27.963 | 29.75 |
| PromptIR V2 | 28.455| 30.27 |
| PromptIR V3 | 29.799 | 30.69 |
| PromptIR V4 | 29.860 | 30.90 |

## Performance snapshot
<img src="img/pubscore.png" width="1000">
