# More Than A Hack 2027 (Phase 1) - Project Log & Documentation

**Repository:** `https://github.com/khizer-flow/tacticalAI.git`  
**Base Upstream:** `https://github.com/metrica-sports/more-than-a-hack-2027.git`  
**Match:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25, Matchday 2  
**Challenge:** Predict frame-by-frame possession (`H`, `A`, `D`, `X`) for 199,319 frames at 25 fps.  
**Primary Metric:** `spellF1` (mean temporal IoU across thresholds 0.3, 0.4, 0.5, 0.6, 0.7).  
**Secondary Tie-break Metric:** `macroF1` (mean per-class F1 for H, A, D).  

---

## 1. Multi-Machine Development Workflow (Laptop & Desktop Sync)

To switch smoothly between laptop and desktop:

### 1.1 Remote Setup
- **`origin`**: `https://github.com/khizer-flow/tacticalAI.git` (Primary working repository for code, scripts, and logs).
- **`upstream`**: `https://github.com/metrica-sports/more-than-a-hack-2027.git` (Official competition base repository for future organizer updates).

### 1.2 Setting Up the Second Machine (e.g. Laptop / Desktop)
1. **Clone your repository:**
   ```bash
   git clone https://github.com/khizer-flow/tacticalAI.git
   cd tacticalAI
   pip install -r requirements.txt
   ```
2. **Link upstream for competition updates:**
   ```bash
   git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git
   ```
3. **Drop the 4 essential tactical data files into `data/tactical/`:**
   Ensure `data/tactical/` contains:
   - `BAR-ATH_tactical_atd.txt`
   - `BAR-ATH_tactical_atd.xml`
   - `BAR-ATH_tactical_homography.csv`
   - `BAR-ATH_tactical_smart_tagging.xml`
4. **Generate & Validate predictions:**
   ```bash
   python starter/improved_possession.py --out evaluation/results/submission_v1.csv
   python evaluation/validate_submission.py evaluation/results/submission_v1.csv
   python evaluation/run_evaluation.py
   ```

### 1.3 Day-to-Day Routine (Preventing Merge Conflicts)
- **Before starting work on either machine:**
  ```bash
  git pull origin main
  ```
- **When finishing a work session:**
  ```bash
  git add .
  git commit -m "Describe updates made"
  git push origin main
  ```

---

## 2. Project Overview & Environment Setup

- **Workspace:** `more-than-a-hack-2027` / `tacticalAI`
- **Data Configuration:**
  - 4 essential tactical data files placed inside `data/tactical/`:
    1. `BAR-ATH_tactical_atd.txt` (195 MB - Tracking coordinates for 46 track slots and the ball)
    2. `BAR-ATH_tactical_atd.xml` (59 KB - Metadata defining teams and track slot mappings)
    3. `BAR-ATH_tactical_smart_tagging.xml` (261 KB - 638 tactical phase & set piece annotations)
    4. `BAR-ATH_tactical_homography.csv` (23 MB - 3x3 camera-to-pitch homography matrices)
  - 13 GB `.mp4` video files skipped as tactical evaluation is purely coordinate & frame-driven.
- **Reference Ground Truth:**
  - Public GT: First 5 minutes of play (`11185` to `18684`, 7,500 frames, 17 true possessions, mean duration 14.2s).
  - Full match: 158,857 graded frames (1st half: 11,185 - 90,651; 2nd half: 110,808 - 190,197).

---

## 3. Baseline Verification & Initial Benchmarks

We executed `evaluation/run_evaluation.py` against the public 5-minute ground truth on the four provided baselines:

| Submission | macroF1 | spellF1 (Public GT) | Possessions (Pred vs 17 GT) | Mean Duration (vs 14.2s GT) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 (-1) | 18.5s (+30%) | Over-smoothed; filters out short possessions |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 (+9) | 10.8s (-24%) | Published full-match reference floor (0.2433 full match) |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6s (-47%) | Frequent micro-fragmentation |
| `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0s (-86%) | Severe flicker; 6.52x fragmentation ratio |

### Baseline Failure Mode Analysis:
1. **Missing Ball Gaps (41% of graded frames):**
   `baseline.py` strictly checks if the ball is visible (`parts[2][0] == MISSING`). When missing, it assigns `NOBODY` (`D`). In reality, ball tracking frequently drops during passes, aerial duels, or crowded boxes. Treating every tracking drop as a stoppage causes false `D` spikes that shatter continuous possessions.
2. **Ignoring Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):**
   The baseline completely ignores the 638 tactical phases and set pieces. Smart tagging explicitly indicates attacking/defending phases (Build Up, Progression, Ball in Final Third, Creating Chances) and clear set pieces (Kick Off, Goal Kick, Corner Kick, Throw-in, Free Kick).
3. **Crudeness of Centered Majority Vote:**
   Uniform sliding-window majority voting is symmetric and blurs transition boundaries (introducing systematic latency) without respecting minimum spell physics. A pass between teammates takes 1-3 seconds, during which neither player is in contact with the ball, yet possession remains with the passing team.

---

## 4. Engineering Improvements Roadmap

1. **Ball Trajectory Imputation & Short-Gap Interpolation:**
   - Detect short gaps (< 1.5s / 35 frames) where the ball track disappears.
   - Interpolate ball location across physical trajectories rather than immediately triggering false dead-ball states.
2. **Possession State Machine & Persistence:**
   - Model possession persistence: a team maintains possession until the opposing team demonstrably controls the ball or play is stopped.
   - Eliminate in-flight pass flickering to opposing defenders.
3. **Smart Tagging Integration:**
   - Map team names (`Barcelona` -> `H`, `Atletic Club` -> `A`).
   - Use tactical phase intervals to reinforce team possession priors.
   - Use field boundaries (`x not in [0, 105]`, `y not in [0, 68]`) and prolonged tracking loss to accurately detect dead-ball `D` states.
4. **Hysteresis Smoothing / Minimum Spell Duration Filtering:**
   - Enforce minimum physical duration threshold (~2.5s / 65 frames) to eliminate micro-flickers that split possessions.

---

## 5. Implementation: `starter/improved_possession.py`

We implemented `starter/improved_possession.py` containing:
- `extract_raw_features`: Single streaming pass over the 199,319 ATD lines calculating `ball_x`, `ball_y`, and minimum distances to Home and Away players (`home_dist`, `away_dist`).
- `interpolate_ball_gaps`: Linear interpolation for missing ball coordinates and team distances across short occlusions ($\le 35$ frames / 1.4s).
- `build_improved_submission`: Implements the possession state machine:
  - **Dead Ball Trigger:** Missing ball exceeding 35 frames or ball moving out of pitch boundaries ($\pm 0.3$m).
  - **Restart Trigger:** Controlled ball re-entry ($d \le 2.5$m).
  - **Live Possession Control:** Control radius $r_{ctrl} = 3.5$m with hysteresis margin $d_{margin} = 1.0$m and minimum switch threshold $sw = 10$ frames.
  - **Smart Tagging Prior:** Attacking tactical phase bonus (+0.2m effective proximity) for active phases (`PROGRESSION`, `BUILD UP`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `CREATING CHANCES`).
  - **Temporal Filter:** `filter_short_spells` merges any transient possession shorter than 65 frames (2.6s) into the surrounding spell.
  - **Compliance:** Outside graded halves (11,185..90,651 and 110,808..190,197), frames are strictly set to `X`.

---

## 6. Results & Benchmarks

### 6.1 Validation Check (`validate_submission.py`)
```bash
python evaluation/validate_submission.py evaluation/results/submission_v1.csv
```
**Output:**
- Rows: `199,319` (0..199,318 contiguous integer frames)
- Valid: **PASSED**
- Label distribution:
  - `H`: 53,415 (26.8%)
  - `A`: 33,639 (16.9%)
  - `D`: 71,803 (36.0%)
  - `X`: 40,462 (20.3% - exact match for ungraded frames)
- Full match possession count: **269 possessions of mean 12.9s** (matches realistic professional football match statistics).

### 6.2 Official Scoreboard Comparison (`run_evaluation.py`)
```bash
python evaluation/run_evaluation.py
```

| Rank | Submission | macroF1 | spellF1 | Possessions (Pred vs 17 GT) | Mean Duration (vs 14.2s GT) | transF1 | Home Share Error | Fragmentation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`submission_v1` (Ours)** | **0.6849** | **0.4444** | **19 (+2)** | **13.7s (-3%)** | **0.3721** | **0.022** | **1.14** |
| 2 | `baseline_smooth125` | 0.5812 | 0.4242 | 16 (-1) | 18.5s (+30%) | 0.1667 | 0.052 | 0.81 |
| 3 | `baseline_smooth50` | 0.6207 | 0.3256 | 26 (+9) | 10.8s (-24%) | 0.0816 | 0.048 | 1.43 |
| 4 | `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6s (-47%) | 0.2034 | 0.056 | 1.90 |
| 5 | `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0s (-86%) | 0.1410 | 0.046 | 6.52 |

### 6.3 Detailed Metric Gains of `submission_v1` over `baseline_smooth125`:
- **`spellF1`:** Improved from **0.4242 to 0.4444** (+2.02 points).
- **`macroF1`:** Improved from **0.5812 to 0.6849** (+10.37 percentage points gain!).
- **Transition F1 (`transF1` @ $\pm 25$ frames):** More than doubled from **0.1667 to 0.3721** (+123% relative improvement).
- **Possession Duration Accuracy:** Duration improved to **13.7s** (only 3% deviation from true 14.2s GT, compared to baseline_smooth125's 18.5s which was 30% over-smoothed).
- **Possession Share Error:** Cut by more than half from **0.0520 down to 0.0221**.
- **Per-Class Dead Ball F1 (`D`):** Surged from **0.1464 to 0.4859** (a 3.3x improvement in dead ball detection accuracy).
