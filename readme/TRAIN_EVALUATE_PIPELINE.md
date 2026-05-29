# Train & Evaluate Pipeline - Quick Start

## Overview

Automated pipeline that:
1. **Trains** the model for 200 episodes (default)
2. **Automatically fetches** the latest run
3. **Evaluates** on test set (30 episodes by default)
4. **Shows summary** of results

---

## Two Ways to Use

### Option A: Python Script (Recommended)

```bash
# Default: 200 training episodes, 30 evaluation episodes
python3 train_and_evaluate.py

# Custom settings
python3 train_and_evaluate.py --episodes 100 --eval-episodes 50

# All options
python3 train_and_evaluate.py --episodes 200 --eval-episodes 30 --log-episodes 1
```

### Option B: Shell Script (Simpler)

```bash
# Default: 200 training episodes, 30 evaluation episodes
./run_full_pipeline.sh

# Custom: 50 training episodes, 20 evaluation episodes
./run_full_pipeline.sh 50 20

# More training: 300 episodes, 50 evaluation
./run_full_pipeline.sh 300 50
```

---

## What Happens

### Step-by-Step Execution

```
1. Activate virtual environment ✓
2. Run training (200 episodes by default) ✓
   └─ Creates: output_data/fake_ppo_run_YYYY-MM-DD_HH-MM-SS/
      ├─ policy_net.pth
      ├─ episode_metrics.csv
      ├─ training_summary.txt
      └─ episode_logs_csv/ (200 files)

3. Find latest run automatically ✓
   └─ No manual path required!

4. Run evaluation (30 episodes by default) ✓
   └─ Creates: evaluation_results/
      ├─ evaluation_results.csv
      └─ evaluation_summary.txt

5. Print results summary ✓
```

---

## Quick Command Reference

| Goal | Command |
|------|---------|
| **Full pipeline (default)** | `python3 train_and_evaluate.py` |
| **Fewer episodes (quick test)** | `python3 train_and_evaluate.py --episodes 50` |
| **More training** | `python3 train_and_evaluate.py --episodes 500` |
| **Different eval size** | `python3 train_and_evaluate.py --eval-episodes 50` |
| **Shell script version** | `./run_full_pipeline.sh` |
| **Shell script with 100 episodes** | `./run_full_pipeline.sh 100` |

---

## Example: Running the Pipeline

```bash
$ python3 train_and_evaluate.py
====================================
🚀 TRAIN & EVALUATE PIPELINE
====================================
Episodes:           200
Log Frequency:      Every 1 episode(s)
Eval Episodes:      30
Initial Balance:    $10,000.00

====================================
STEP 1: TRAINING
====================================
Training for 200 episodes...

Episode 1: Reward=0.0010 | Net Worth=$10,023.45 | ...
Episode 2: Reward=0.0025 | Net Worth=$10,045.67 | ...
...
Episode 200: Reward=0.1234 | Net Worth=$11,234.50 | ...
✅ Training completed successfully!

====================================
STEP 2: EVALUATION
====================================
Latest run: fake_ppo_run_2026-05-20_10-42-15
Evaluating on test set (30 episodes)...

Eval  1 | stock1       | Return:  +5.23% | ...
Eval  2 | stock2       | Return: +12.45% | ...
...
Eval 30 | stock30      | Return:  +8.67% | ...
✅ Evaluation completed successfully!

====================================
RESULTS SUMMARY
====================================
📊 Test Performance:
  Win Rate:              80.0%
  Mean Return:           +8.45%
  Max Drawdown:          -5.23%

📈 Train vs Test:
  Mean Return:  Train: +7.50% → Test: +8.45%
  Win Rate:     Train: 75.0% → Test: 80.0%

📁 RESULTS SAVED TO:
Training Results:   output_data/fake_ppo_run_2026-05-20_10-42-15/
Evaluation Results: output_data/fake_ppo_run_2026-05-20_10-42-15/evaluation_results/

✅ PIPELINE COMPLETE!
```

---

## Output Files

After running the pipeline, you'll have:

```
output_data/fake_ppo_run_2026-05-20_10-42-15/
├── policy_net.pth                          ← Trained model
├── episode_metrics.csv                     ← Training metrics (200 rows)
├── training_summary.txt                    ← Training stats
├── training_rewards.csv                    ← Rewards over episodes
├── episode_logs_csv/
│   ├── episode_1_debug_log.csv
│   ├── episode_2_debug_log.csv
│   └── ... (200 files)
├── episode_logs_png/
│   ├── episode_1_performance.png
│   ├── episode_2_performance.png
│   └── ... (200 files)
└── evaluation_results/
    ├── evaluation_results.csv              ← Test metrics (30 rows)
    └── evaluation_summary.txt              ← Test stats + comparison
```

---

## Advanced Usage

### Monitor Training in Dashboard

While training, in a **separate terminal**:

```bash
cd /home/gayathri/min\ proj2/rl-stock-intraday-project
source .venv/bin/activate
streamlit run dashboard.py
```

Then open: `http://localhost:8501`

The dashboard will show live updates:
- Episode metrics updating every 2 seconds
- Equity curves for each episode
- Risk metrics and distributions
- Q-value analysis

### Custom Training Parameters

```bash
# Different learning rate
python3 train_and_evaluate.py --episodes 200  # (calls train_chronological.py internally)

# To pass custom training params, use train_chronological.py directly:
python3 train_chronological.py --episodes 200 --learning-rate 0.001 --batch-size 32
python3 evaluate_model.py --run-dir <latest-run> --eval-episodes 30
```

### Just Evaluate (without training)

```bash
python3 evaluate_model.py --run-dir output_data/fake_ppo_run_2026-05-20_08-05-51 --eval-episodes 50
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
# Activate venv first
source .venv/bin/activate
python3 train_and_evaluate.py
```

### Issue: "No training run found"
Make sure training completes successfully (check for `policy_net.pth` in the output directory)

### Issue: Permission denied for shell script
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## Comparison: Before vs After

| Task | Before | After |
|------|--------|-------|
| **Run training** | `python3 train_chronological.py` | `python3 train_and_evaluate.py` |
| **Find latest run** | Manually: `ls output_data/` | Automatic ✓ |
| **Run evaluation** | Separate command with manual path | Automatic ✓ |
| **View results** | Manual file browsing | Print summary ✓ |
| **Total steps** | 3+ commands | 1 command |

---

## Recommended Workflow

```bash
# Quick test (5 min)
python3 train_and_evaluate.py --episodes 20

# Standard run (20-30 min)
python3 train_and_evaluate.py

# Thorough run (1+ hour)
python3 train_and_evaluate.py --episodes 500 --eval-episodes 50

# While training, monitor in another terminal
streamlit run dashboard.py
```

