# Quick Reference: Updated Visualization Files

## What Changed?

### Problem
The visualization files were hardcoded for **binary actions** (0 = Hold, 1 = Buy). The new agent uses **continuous actions** from -1.0 to +1.0, and the dashboard crashed with `KeyError: 'Action_Value'`.

### Solution
Made both files **format-agnostic** - they auto-detect which format the CSV has and display accordingly.

---

## Dashboard Sections

### A — Live Episode Monitor
Current episode stats: Net Worth, Return %, Trades, etc.

### **B.5 — Latest Step: Action & Q-Values** ← NEW!
Shows the most recent action decision:
- **Latest Action**: Human-readable format like "BUY 70%" or "SELL 50%"
- **Chosen Q-Value**: The actual Q-value that was selected
- **Current Price**: Stock price at this step
- **Shares Held**: Position size
- **Q-Values Preview**: All available Q-values for this step

Example output:
```
Latest Action        Chosen Q-Value       Current Price       Shares Held
BUY 70%             7.4512               $1,387.75           2
```

### B — Live Equity Curve
Net worth evolution with Buy/Sell/Hold markers overlaid on price chart.

### C — Risk Metrics Panel
Drawdown, volatility, Sharpe ratio, exploration %, Q-values.

### D & E — Distributions & Evaluation
Return distribution, trade count distribution, evaluation performance.

---

## Data Column Detection

The code automatically checks for:

**New Format (has `Action_Value`):**
- Uses continuous values (-1.0 to +1.0)
- Displays Buy/Sell/Hold
- Shows percentage positions (e.g., "BUY 70%")
- Uses Q_Values_Preview column

**Old Format (has `Action` but no `Action_Value`):**
- Uses binary values (0 = Hold, 1 = Buy)
- Only shows Buy/Hold (no Sell)
- Converts to continuous for display
- Uses Q_Cash and Q_Invest columns

---

## Action Text Formatting

```python
_format_action_text(0.7)   →  "BUY 70%"
_format_action_text(-0.5)  →  "SELL 50%"
_format_action_text(0.05)  →  "HOLD"
```

**Thresholds:**
- < -0.1: SELL
- -0.1 to 0.1: HOLD
- > 0.1: BUY

---

## Running the Dashboard

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

**Auto-detects latest run** and refreshes every 2 seconds.

---

## Files Updated

1. **dashboard.py** (562 → 610 lines)
   - Added format detection
   - Added action text display
   - Added Q-values display section

2. **advanced_analytics.py** (623 → 678 lines)
   - Updated stats computation
   - Updated plot generation
   - Added format detection

3. **performance_commentary.py**
   - No changes needed

---

## Error Handling

✓ Missing `Action_Value` column → Uses `Action` (binary format)
✓ Missing Q-value columns → Shows "N/A"
✓ Empty DataFrames → Shows informational message
✓ Malformed Q_Values_Preview → Returns empty dict

All errors are caught gracefully!

---

## Testing Checklist

- [x] Syntax validation: Both files compile without errors
- [x] Backward compatibility: Works with old binary action format
- [x] Forward compatibility: Works with new continuous action format
- [x] Column detection: Auto-detects both formats correctly
- [x] Error handling: Gracefully handles missing columns
- [x] Dashboard rendering: Streamlit app starts and displays data
- [x] Action text display: Shows "BUY X%", "SELL X%", "HOLD" correctly
- [x] Q-values display: Shows both single value and preview string
- [x] Helper functions: All new functions defined and used

---

## Example Output

When you run the dashboard with new format data:

```
Episode 240 | Trading Stock: itc
Episode 240: Reward=0.21 | Net Worth=$11,234.50 | Return=12.35% | Trades=47

Latest Action: BUY 70%
Value: 0.7000
Chosen Q-Value: 7.6542
Current Price: $1,387.75
Shares Held: 2
Q-Values Preview: 7.4772, 7.4948, 7.5023, 7.4962, ...
```

---

## Future Improvements

- [ ] Add trade history timeline
- [ ] Export action log to CSV
- [ ] Add confidence/uncertainty visualization
- [ ] Show profit per trade metrics
- [ ] Add portfolio rebalancing view
