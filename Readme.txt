Intraday Stock Trading Agent (DQN)
---------------------------------------
#How to Run

1️⃣ Install Dependencies
    pip install torch pandas numpy matplotlib gymnasium

2️⃣ Make Sure Data Exists
    Your folder must contain:
        processed_data/
            ├── stock1-minute.csv
            ├── stock2-minute.csv
            └── ...

3️⃣ Run Training
    Basic run:
        python train_simplified.py
    
    Run with custom parameters:
        python train_simplified.py --episodes 200 

📂 Output Structure
    output_data/
    └── run_YYYY-MM-DD_HH-MM-SS/
            ├── trained_model_simplified.pth
            ├── episode_metrics.csv
            ├── training_rewards/
            │       ├── training_rewards_ep50.png
            │       └── training_rewards_final.png
            ├── episode_logs_csv/
            │       ├── episode_10_debug_log.csv
            │       └── ...
            └── episode_logs_png/
                    ├── episode_10_performance.png
                    └── ...
        
What Each Output Contains
🔹 trained_model_simplified.pth
        Saved PyTorch model weights
🔹 episode_metrics.csv
        Per-episode summary including:
            >Total Reward
            >Final Net Worth
            >Percentage Return
            >Number of Trades
🔹 training_rewards/
        Graphs showing reward progression over episodes.
🔹 episode_logs_csv/
        Detailed step-by-step logs for selected episodes:
            >Step number
            >Action taken
            >Q-values
            >Reward
            >Net worth
            >Exploration vs exploitation

🔹 episode_logs_png/
        Performance visualization for logged episodes:
            >Net worth curve
            >Buy/Sell markers
            >Price curve