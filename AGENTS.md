# AGENTS.md

Welcome! If you are an AI coding agent working on this repository, please read **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** immediately.

## Quick Reference
- **Mission:** Predict possession (`H`, `A`, `D`, `X`) for 199,319 frames on FC Barcelona vs Athletic Club (`BAR-ATH`).
- **Primary Metric:** `spellF1` (Current Best: **`0.5371`** via `submission_v3.csv`, Rank #1).
- **Secondary Metric:** `macroF1` (Current Best: **`0.7592`**).
- **Production Engine:** `starter/improved_possession.py` (Supports `--version v1`, `--version v2`, `--version v3`).
- **Diagnostic Tool:** `starter/inspect_errors.py` (Compares predictions against all 21 Ground Truth segments).
- **Project Log:** `PROJECT_LOG.md` (Living history of all experiments, formulas, and baseline results).

### Commands
```bash
# Run SOTA Version 3
python starter/improved_possession.py --version v3 --out evaluation/results/submission_v3.csv

# Validate submission
python evaluation/validate_submission.py evaluation/results/submission_v3.csv

# Run evaluation scoreboard
python evaluation/run_evaluation.py

# Run error diagnostics
python starter/inspect_errors.py
```

See **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** for detailed architecture, mathematical rules, and v4 roadmap.
