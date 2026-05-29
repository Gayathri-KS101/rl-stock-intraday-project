# Quick Reference: Train & Evaluate Workflow

## One-Command Cheat Sheet

```bash
# 1. TRAIN (takes 5-30 minutes depending on episodes)
python3 train_chronological.py --episodes 200

# 2. FIND YOUR RUN (copy the directory name)
ls -la output_data/ | grep fake_ppo

# 3. EVALUATE (test on 30% hold-out data)
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08

# 4. VIEW RESULTS
cat output_data/fake_ppo_run_2026-05-19_19-54-08/evaluation_results/evaluation_summary.txt
```

---

## Data Split

```
All Trading Days (sorted by date)
    ↓
[70% TRAINING] | [30% TESTING]
    ↓               ↓
Train Model    Evaluate Model
                (this is new!)
```

---

## Directory Structure After Training

```
output_data/
└── fake_ppo_run_2026-05-19_19-54-08/
    ├── policy_net.pth              ← Trained model
    ├── episode_metrics.csv         ← Training results
    ├── training_summary.txt        ← Training stats
    ├── episode_logs_csv/           ← Detailed logs
    ├── episode_logs_png/           ← Visualizations
    └── evaluation_results/         ← NEW! (after evaluate_model.py)
        ├── evaluation_results.csv
        └── evaluation_summary.txt
```

---

## Key Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **Win Rate** | % episodes with profit | > 50% |
| **Mean Return** | Average per-episode return | > 0% |
| **Test vs Train** | Generalization check | Test ≥ Train |
| **Max Drawdown** | Largest loss | > -20% |

---

## Common Issues & Solutions

### Issue: "run-dir not found"
```bash
# Solution: Copy full path from ls output
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-19_19-54-08
```

### Issue: "Model not found"
```bash
# Solution: Make sure policy_net.pth exists in the run directory
ls output_data/fake_ppo_run_2026-05-19_19-54-08/policy_net.pth
```

### Issue: "Not enough test data"
```bash
# Solution: Use more episodes (get more diverse test cases)
python3 evaluate_model.py --run-dir <path> --eval-episodes 50
```

---

## Customize Commands

```bash
# Train with fewer episodes (quick test)
python3 train_chronological.py --episodes 10

# Train with different split (80/20)
python3 train_chronological.py --train-ratio 0.8

# Evaluate with more test episodes
python3 evaluate_model.py --run-dir <path> --eval-episodes 50

# Evaluate with different starting balance
python3 evaluate_model.py --run-dir <path> --initial-balance 50000

# All together
python3 train_chronological.py --episodes 100 --train-ratio 0.75
python3 evaluate_model.py --run-dir <latest> --eval-episodes 25 --initial-balance 10000
```

---

## Interpretation Guide

### If Test Performance > Training Performance ✅
```
Mean Return: Train +1.5% → Test +2.3%
↓
✅ GOOD! Model generalizes well to unseen data
```

### If Test Performance < Training Performance ⚠️
```
Mean Return: Train +3.0% → Test +1.2%
↓
⚠️ WARNING! Model may be overfitting
Suggestion: Use more training data or reduce model complexity
```

### If Test Performance ≈ Training Performance ✓
```
Mean Return: Train +2.1% → Test +2.0%
↓
✓ CONSISTENT! Model performs similarly on train and test
```

---

## Files You'll Need

- `evaluate_model.py` - Evaluation script
- `train_chronological.py` - Training script (unchanged, just run it)
- `policy_net.pth` - Generated after training (in output_data/)

---

## Monitoring During Training

While training, monitor in a separate terminal:

```bash
# Terminal 1: Training
python3 train_chronological.py --episodes 200

# Terminal 2: Live dashboard
streamlit run dashboard.py
```

After training + evaluation, you'll see:
- Training metrics on dashboard
- Test metrics in evaluation_results/

---

## Full Workflow (Step by Step)

```bash
# Step 1: Train (choose your settings)
python3 train_chronological.py --episodes 100 --log-episodes 5

# ⏳ Wait for completion (5-30 min depending on episodes)

# Step 2: Find run directory
RUN_DIR=$(ls -dt output_data/fake_ppo_run_* | head -1)
echo "Latest run: $RUN_DIR"

# Step 3: Evaluate
python3 evaluate_model.py --run-dir $RUN_DIR --eval-episodes 30

# Step 4: View training stats
cat $RUN_DIR/training_summary.txt

# Step 5: View test results
cat $RUN_DIR/evaluation_results/evaluation_summary.txt

# Step 6: Compare metrics
echo "=== TRAINING METRICS ===" && head -5 $RUN_DIR/episode_metrics.csv
echo "=== TEST METRICS ===" && head -5 $RUN_DIR/evaluation_results/evaluation_results.csv
```

---

## What's Different Now?

### Before ❌
- Train on 70% data
- Test data loaded but ignored
- No way to check if model generalizes

### Now ✅
- Train on 70% data (same)
- **Evaluate on 30% test data** (NEW!)
- **See train vs test comparison** (NEW!)
- **Detect overfitting** (NEW!)

