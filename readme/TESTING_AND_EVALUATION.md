# Testing & Evaluation Guide

## Overview

The project uses a **chronological train/test split** to prevent temporal leakage:
- **70% of trading days** → Training (model learns on these)
- **30% of trading days** → Testing (held out, used for evaluation)

**Currently:** ❌ **NO automatic testing** happens after training completes  
**Now Available:** ✅ **New evaluation script** to test on 30% hold-out data

---

## Workflow

### Step 1: Train the Model
```bash
python3 train_chronological.py --episodes 200
```

**Output:**
- `output_data/fake_ppo_run_2026-05-19_19-54-08/`
  - `policy_net.pth` (trained model)
  - `episode_metrics.csv` (training performance)
  - `training_summary.txt` (training stats)
  - `episode_logs_csv/` (detailed logs)
  - `episode_logs_png/` (visualizations)

### Step 2: Evaluate on Test Set (NEW!)
```bash
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08 --eval-episodes 30
```

**Output:**
- `evaluation_results/`
  - `evaluation_results.csv` (per-episode metrics)
  - `evaluation_summary.txt` (stats)
  - Compare test vs training performance

---

## Data Split Details

### How Splitting Works
The `data_splitter.py` splits data **by trading days** (not by individual minutes):

```
Input: All trading days (chronologically sorted)
        ├─ Day 1 data
        ├─ Day 2 data
        ├─ Day 3 data
        └─ ...Day N data

Split at 70%:
        TRAIN (70% days)                TEST (30% days)
        ├─ Day 1 data                   ├─ Day (0.7N+1) data
        ├─ Day 2 data          ----→    ├─ Day (0.7N+2) data
        └─ ...Day 0.7N data             └─ ...Day N data
```

### Why This Prevents Leakage
✅ **No temporal leakage** - Test data is strictly AFTER training data  
✅ **Real-world simulation** - Model evaluates on unseen, future trading days  
✅ **Fair comparison** - Training and testing on completely separate time periods

---

## Commands

### Basic Training
```bash
# Train with default settings (200 episodes)
python3 train_chronological.py
```

### Custom Training
```bash
# Train fewer episodes
python3 train_chronological.py --episodes 50

# Change learning rate
python3 train_chronological.py --episodes 200 --learning-rate 0.001

# Change train/test ratio
python3 train_chronological.py --train-ratio 0.8  # 80% train, 20% test

# Log every episode (default now, creates 200 plots)
python3 train_chronological.py --log-episodes 1

# Log every 5 episodes (saves disk space)
python3 train_chronological.py --log-episodes 5
```

### Basic Evaluation
```bash
# Evaluate latest run
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08
```

### Custom Evaluation
```bash
# Run 50 test episodes (instead of default 30)
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08 --eval-episodes 50

# Test with different starting balance
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08 --initial-balance 50000
```

---

## Understanding Evaluation Results

### Key Metrics

| Metric | Meaning | Good Range |
|--------|---------|-----------|
| **Win Rate** | % of test episodes with positive return | > 50% |
| **Mean Return** | Average return per episode | > 0% |
| **Std Dev Return** | Return volatility | Lower is more stable |
| **Max Drawdown** | Largest peak-to-trough decline | > -10% |
| **Mean Trades** | Average trades per episode | Domain specific |

### Example Output

```
TEST SET PERFORMANCE:
  Episodes:              30
  Win Rate:              56.7%           ✅ Better than 50%
  Mean Return:           +2.34%          ✅ Positive
  Std Dev Return:        3.45%           ✓ Reasonable
  Min Return:            -8.50%
  Max Return:            +12.30%
  Median Return:         +1.95%
  Mean Max Drawdown:     -4.23%          ✓ Reasonable
  Mean Trades/Episode:   15.3

TRAIN vs TEST COMPARISON:
  Mean Return:           Train: +2.10% → Test: +2.34% (+0.24%)
  Win Rate:              Train: 52.0% → Test: 56.7% (+4.7%)

✅ Test performance is BETTER than training (good generalization)
```

### Interpretation Guide

#### Good Signs ✅
- Test win rate > 50%
- Test mean return > 0%
- Test performance ≥ training performance
- Consistent returns (low std dev)
- Max drawdown < -20%

#### Warning Signs ⚠️
- Test win rate < 50%
- Test mean return < 0%
- Test performance << training performance (overfitting)
- High volatility (high std dev)
- Max drawdown > -30%

---

## Complete Workflow Example

```bash
# Step 1: Train the model
python3 train_chronological.py --episodes 200 --log-episodes 5

# Wait for training to complete (~20-30 minutes)
# Monitor in Streamlit dashboard:
#   streamlit run dashboard.py

# Step 2: Evaluate on test set
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08 --eval-episodes 30

# Step 3: View results
# Open: output_data/fake_ppo_run_2026-05-19_19-54-08/evaluation_results/evaluation_summary.txt
# View: output_data/fake_ppo_run_2026-05-19_19-54-08/evaluation_results/evaluation_results.csv
```

---

## What Each Output File Contains

### Training Outputs
- **episode_metrics.csv**: Per-episode training results (reward, return, trades)
- **training_summary.txt**: Training configuration and statistics
- **policy_net.pth**: Model weights (can be loaded later)
- **episode_logs_csv/**: Step-by-step logs for each episode
- **episode_logs_png/**: Visualizations of equity curves, trades, Q-values

### Evaluation Outputs (in `evaluation_results/`)
- **evaluation_results.csv**: Per-episode test results
- **evaluation_summary.txt**: Summary statistics and train/test comparison
- **Returns distribution, drawdown metrics**: Comprehensive test analysis

---

## Finding Your Run Directory

After training, find your run in:
```bash
ls -la output_data/ | grep fake_ppo_run
```

Output:
```
drwxr-xr-x  fake_ppo_run_2026-05-19_19-54-08
drwxr-xr-x  fake_ppo_run_2026-05-19_19-30-22
drwxr-xr-x  fake_ppo_run_2026-05-19_18-45-00
```

Use the **most recent** one (or the one you want to evaluate):
```bash
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08
```

---

## Frequently Asked Questions

**Q: Does training automatically test on 30% of data?**  
A: ❌ No, currently training only uses 70% data. You must manually run `evaluate_model.py` after training.

**Q: How do I know if my model is overfitting?**  
A: Compare train vs test metrics in evaluation summary:
- If test performance << training → **overfitting**
- If test performance ≈ training → **good generalization**
- If test performance > training → **excellent!**

**Q: Can I evaluate the same model multiple times?**  
A: ✅ Yes, run `evaluate_model.py` as many times as you want. It won't affect the trained model.

**Q: What if I want to test with different settings?**  
A: All test settings (number of episodes, starting balance) can be customized:
```bash
python3 evaluate_model.py --run-dir <path> --eval-episodes 50 --initial-balance 50000
```

**Q: Where should I check results?**  
A: Check these files in `evaluation_results/`:
1. **evaluation_summary.txt** - Human-readable summary
2. **evaluation_results.csv** - Detailed per-episode metrics
3. Compare with **episode_metrics.csv** in parent directory (training metrics)

---

## Suggested Workflow for Development

```bash
# 1. Quick test with small dataset
python3 train_chronological.py --episodes 20 --log-episodes 1

# 2. Evaluate immediately
python3 evaluate_model.py --run-dir <latest-run> --eval-episodes 20

# 3. If results look good, train full model
python3 train_chronological.py --episodes 200 --log-episodes 5

# 4. Full evaluation
python3 evaluate_model.py --run-dir <latest-run> --eval-episodes 30
```

