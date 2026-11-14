# Agent Handoff & Quickstart Guide

> **Welcome Coding Agent / Contributor!**  
> This file is purpose-built to give you 100% of the context, reproduction commands, codebase architecture, and next optimization targets in 60 seconds so you can immediately begin working without having to discover everything from scratch.

---

## 1. Executive Summary & Current Leaderboard Standing

- **Challenge:** More Than A Hack 2027 (Phase 1)
- **Task:** Predict frame-by-frame match possession (`H` for Barcelona, `A` for Athletic Club, `D` for Dead Ball, `X` for ungraded) for **199,319 contiguous frames** at 25 fps.
- **Primary Competition Metric:** `spellF1` (Mean temporal IoU at thresholds 0.3, 0.4, 0.5, 0.6, 0.7).
- **Secondary Tie-Breaker Metric:** `macroF1` (Mean unweighted F1 over classes H, A, D).
- **Official Public Ground Truth:** First 5 minutes of play (`11,185` to `18,684`, 7,500 frames, 17 true possessions, 4 stoppages, mean duration 14.2s).
- **Current Standing:** **RANK #1** across all models!

### Official Benchmark Table (`python evaluation/run_evaluation.py`):
| Rank | Submission | macroF1 | **spellF1** | Possessions (Pred vs 17 GT) | Mean Duration (vs 14.2s GT) | transF1 | Mean Latency | Share Error | Fragmentation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`submission_v3` (Current Best)** | **0.7592** | **0.5371** | **18 (+1)** | **14.2s (0.0s error!)** | **0.1905** | **`-1.2` frames** | **0.0031** | **1.10** |
| **2** | **`submission_v2`** | **0.7293** | **0.4667** | **19 (+2)** | **12.9s (-9%)** | **0.3256** | **`-9.7` frames** | **0.0012** | **1.14** |
| **3** | **`submission_v1`** | **0.6849** | **0.4444** | **19 (+2)** | **13.7s (-3%)** | **0.3721** | **`-9.8` frames** | **0.0221** | **1.14** |
| 4 | `baseline_smooth125` | 0.5812 | 0.4242 | 16 (-1) | 18.5s (+30%) | 0.1667 | `-7.7` frames | 0.0520 | 0.81 |
| 5 | `baseline_smooth50` | 0.6207 | 0.3256 | 26 (+9) | 10.8s (-24%) | 0.0816 | `+4.0` frames | 0.0485 | 1.43 |
| 6 | `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6s (-47%) | 0.2034 | `+8.2` frames | 0.0556 | 1.90 |
| 7 | `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0s (-86%) | 0.1410 | `-3.1` frames | 0.0457 | 6.52 |

---

## 2. Fast Commands for Coding Agents

### Step 1: Run the Inference Engine
```bash
# Runs the state-of-the-art Version 3 model (takes ~4.5 seconds for all 199,319 frames)
python starter/improved_possession.py --version v3 --out evaluation/results/submission_v3.csv
```

### Step 2: Validate the Submission File Format
```bash
python evaluation/validate_submission.py evaluation/results/submission_v3.csv
```
*Validation checks: exactly 199,319 contiguous integer rows, valid column names `frame,label`, valid labels `H,A,D,X`, outside graded frames strictly `X`.*

### Step 3: Run the Official Evaluation Scorer
```bash
python evaluation/run_evaluation.py
```
*Evaluates all CSV submissions in `evaluation/results/` against the ground truth and updates `evaluation/scoreboard.csv`.*

### Step 4: Run Segment-by-Segment Diagnostics
```bash
# Diagnoses prediction alignment vs the 21 GT segments (IoU, duration, and error modes)
python starter/inspect_errors.py
```

---

## 3. Data & Environment Setup

- **Remote Working Repo:** `https://github.com/khizer-flow/tacticalAI.git` (`origin/main`)
- **Base Competition Repo:** `https://github.com/metrica-sports/more-than-a-hack-2027.git` (`upstream`)
- **Required Data Files:** Only **4 small text/XML files (~150 MB total)** in `data/tactical/`:
  1. `data/tactical/BAR-ATH_tactical_atd.txt` (Coordinate tracking for 46 player slots and the ball)
  2. `data/tactical/BAR-ATH_tactical_atd.xml` (Player metadata & team ID mapping)
  3. `data/tactical/BAR-ATH_tactical_homography.csv` (3x3 camera-to-pitch homography matrices)
  4. `data/tactical/BAR-ATH_tactical_smart_tagging.xml` (638 tactical phase & set-piece annotations)
- **Video `.mp4` and `broadcast/` files:** Skipped! They are 13 GB and not synchronized to tactical frames. Scoring is 100% tactical coordinate-based.
- **Git Caution:** Match data files (`data/*`) and generated results (`evaluation/results/*.csv`) are gitignored. Never force-commit raw tracking data.

---

## 4. Key File Architecture

```text
more-than-a-hack-2027/
├── starter/
│   ├── improved_possession.py    # Production inference engine (v1, v2, v3)
│   ├── inspect_errors.py         # Side-by-side GT diagnostic tool
│   ├── load_atd.py               # ATD coordinate loader
│   ├── load_smart_tagging.py     # SportsCode XML event parser
│   └── baseline.py               # Official benchmark baseline
├── evaluation/
│   ├── gt/
│   │   └── BAR-ATH_possession_gt_public.csv  # 5-min public GT (11185..18684)
│   ├── results/
│   │   ├── submission_v1.csv     # v1 predictions (spellF1 = 0.4444)
│   │   ├── submission_v2.csv     # v2 predictions (spellF1 = 0.4667)
│   │   └── submission_v3.csv     # v3 predictions (spellF1 = 0.5371)
│   ├── possession_metrics.py     # Official evaluation metrics & IoU logic
│   ├── run_evaluation.py         # Scoreboard runner
│   └── validate_submission.py    # Strict submission validator
├── games.py                      # Game configuration & asset paths
├── PROJECT_LOG.md                # Full persistent project log & experiment history
└── AGENT_HANDOFF.md              # THIS FILE (Agent instruction & context guide)
```

---

## 5. Algorithmic Evolution & Mathematical Formulations

### Version 1 (`v1`) - Baseline Extension
- **Gap Interpolation:** Linear interpolation for missing ball coordinates across short gaps ($\le 35$ frames / 1.4s).
- **Hysteresis Smoothing:** Filters out micro-spells $< 65$ frames (2.6s).
- **Tactical Priors:** +0.2m effective distance bonus for attacking tactical phases in `smart_tagging`.

### Version 2 (`v2`) - Kinematics & Aerial Physics
- **Directional Velocity Cosine Similarity:**
  $$\cos \alpha = \frac{\vec{v}_{\text{ball}} \cdot \vec{u}_{\text{player}}}{\|\vec{v}_{\text{ball}}\| \|\vec{u}_{\text{player}}\|}$$
  When $v_{\text{ball}} > 5.0\text{ m/s}$, players moving away from the ball trajectory ($\cos \alpha < -0.2$) incur an effective distance penalty:
  $$d_{\text{eff}} = d + \min(3.0, 0.25 \cdot v_{\text{ball}} \cdot (-\cos \alpha))$$
  *Solves the "pressing defender trap" where an opponent sprinting past a pass was closer in Euclidean distance.*
- **Velocity-Gated Control Ceiling:** When $v_{\text{ball}} > 11.5\text{ m/s}$, opponents cannot steal possession mid-flight.
- **In-Bounds Aerial Occlusion:** High punts/clearances blend into stadium sunlight for 3.0 to 4.5s. If departure and arrival are in-bounds ($x \in [4, 101], y \in [3, 65]$), gaps up to 115 frames (4.6s) are interpolated without triggering false dead balls.
- **Set-Piece Pre-Kick Hold:** Holds Dead Ball (`D`) during Free Kicks, Corners, and Goal Kicks until $v_{\text{ball}} > 3.5\text{ m/s}$. Kickoffs have a 50-frame grace period.

### Version 3 (`v3`) - State-of-the-Art Production Engine
- **Abandoned Ball Stoppage Detection:** When a ball is on the grass but unattended by all players ($\min(d_H, d_A) > 4.5\text{m}$) at low speed ($v_{\text{ball}} < 4.5\text{ m/s}$) for $\ge 35$ consecutive frames, the engine flags a Dead Ball (`D`). Captures true referee stoppages like Segment 15 (`03:08` - `03:28`).
- **Bidirectional Anchor Bridging:** Identifies high-confidence controlled touch anchors ($d_{\text{raw}} \le 2.0\text{m}$, $v_{\text{ball}} < 4.0\text{ m/s}$). Gaps $\le 100$ frames (4.0s) between anchors of the same team are bridged as continuous possession, eliminating fragmented passing sequences.
- **Responsive Duel Kinematics:** Tuned turnover margin to $0.8\text{m}$ and switch latency to $6$ frames ($0.24\text{s}$), bringing mean transition latency down to **`-1.2` frames** (virtually instantaneous tracking).
- **Fine-Tuned Minimum Spell Duration:** Hysteresis filter set to $50$ frames ($2.0\text{s}$), preserving legitimate short possessions while preventing high-frequency fragmentation.

---

## 6. Execution Rules for AI Coding Agents

1. **OS Environment:** Windows (PowerShell).
2. **Avoid Inline PowerShell Quote Escaping:** Writing complex one-line Python commands with nested quotes in PowerShell often triggers syntax errors. Instead, write short scratch scripts to `<artifactDir>/scratch/` and execute them via `python <path>`.
3. **Sandbox & Elevated Permissions:**
   - Standard Sandbox Mode is fine for read-only tracking and running Python tests.
   - For `git commit`, `git push`, network access, or writing directly to `evaluation/scoreboard.csv`, use elevated mode (`BypassSandbox: true` in tools).
4. **Preserve Reproducibility:**
   - Do NOT break `v1`, `v2`, or `v3`. Any new algorithm should be added as `--version v4` in `starter/improved_possession.py`.
   - Always run `python evaluation/validate_submission.py <path>` before scoring.
   - Output CSVs must have exactly 199,319 rows.

---

## 7. Actionable Roadmap for Version 4 (`v4`)

Analysis from `inspect_errors.py` reveals the following highest-value opportunities to push `spellF1` from **0.5371 towards 0.60+**:

1. **Contested 50/50 Duel Buffer (`01:00` - `01:04`):**
   - In segment 5/6, an aerial ball drops between two players. Neither player has clean possession for ~1.5s during the header/scramble. Introducing a contested-duel buffer prevents premature state switches before settled control ($v_{\text{ball}} < 4.0\text{ m/s}$ and $d \le 2.0\text{m}$ for $\ge 10$ frames).
2. **Camera Homography Integration (`BAR-ATH_tactical_homography.csv`):**
   - The pitch boundary in meters ($105 \times 68$) has small camera calibration distortions near corners and touchlines. Projecting the coordinates through the homography matrix will sharpen touchline dead-ball transitions.
3. **Smart Tagging Phase-Boundary Fusion:**
   - When Smart Tagging notes the end of an `ATTACKING TRANSITION` or `BALL IN FINAL THIRD` followed by a set piece, snap the Dead Ball start frame to the phase end frame to eliminate small boundary latency.
