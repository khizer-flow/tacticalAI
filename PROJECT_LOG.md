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
| **1** | **`submission_v2` (Ours)** | **0.7293** | **0.4667** | **19 (+2)** | **12.9s (-9%)** | **0.3256** | **0.0012** | **1.14** |
| **2** | **`submission_v1` (Ours)** | **0.6849** | **0.4444** | **19 (+2)** | **13.7s (-3%)** | **0.3721** | **0.0221** | **1.14** |
| 3 | `baseline_smooth125` | 0.5812 | 0.4242 | 16 (-1) | 18.5s (+30%) | 0.1667 | 0.0520 | 0.81 |
| 4 | `baseline_smooth50` | 0.6207 | 0.3256 | 26 (+9) | 10.8s (-24%) | 0.0816 | 0.0485 | 1.43 |
| 5 | `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6s (-47%) | 0.2034 | 0.0556 | 1.90 |
| 6 | `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0s (-86%) | 0.1410 | 0.0457 | 6.52 |

### 6.3 Detailed Metric Gains of `submission_v1` over `baseline_smooth125`:
- **`spellF1`:** Improved from **0.4242 to 0.4444** (+2.02 points).
- **`macroF1`:** Improved from **0.5812 to 0.6849** (+10.37 percentage points gain!).
- **Transition F1 (`transF1` @ $\pm 25$ frames):** More than doubled from **0.1667 to 0.3721** (+123% relative improvement).
- **Possession Duration Accuracy:** Duration improved to **13.7s** (only 3% deviation from true 14.2s GT, compared to baseline_smooth125's 18.5s which was 30% over-smoothed).
- **Possession Share Error:** Cut by more than half from **0.0520 down to 0.0221**.
- **Per-Class Dead Ball F1 (`D`):** Surged from **0.1464 to 0.4859** (a 3.3x improvement in dead ball detection accuracy).

---

## 7. Version 2 Upgrade: Video Kinematics & Aerial Physics (`submission_v2`)

### 7.1 Diagnostic Root Cause from User Video Annotations
By inspecting tactical match footage (specifically match opening `00:00` - `01:15` corresponding to video timestamps `7:28` - `8:43`), we diagnosed the key physical failure modes of naive Euclidean distance tracking:

1. **The Pressing Defender Proximity Trap (`00:01` - `00:11`):**
   - *Failure:* When a player kicks a pass or clearance, a sprinting pressing defender is often closer in Euclidean distance ($d \approx 1.0\text{m}$) than the kicker ($d \approx 2.5\text{m}$) for ~5–15 frames right after the kick. This caused false turnover flickers.
   - *Solution (Directional Cosine Penalty):* When $v_{\text{ball}} > 5.0\text{ m/s}$, we compute the cosine similarity between the ball trajectory vector $\vec{v}_{\text{ball}}$ and the relative displacement vector to the player $\vec{u}_{\text{player}}$. If $\cos \alpha < -0.2$ (the ball is moving rapidly away from the player), an effective distance penalty is applied, preventing spurious turnover triggers.
2. **Velocity-Gated Control Ceiling:**
   - *Failure:* A defender cannot instantaneously take controlled possession of a ball flying past their head at high speed ($v_{\text{ball}} > 11.5\text{ m/s}$).
   - *Solution:* Changes of possession are suppressed while the ball velocity exceeds 11.5 m/s unless settled control is demonstrated.
3. **In-Bounds Aerial Occlusion Handling (`00:03`):**
   - *Failure:* High aerial clearances and punts fly across the stadium roof and harsh sunlight strips, vanishing from the 2D tracking system for 3.0 to 4.5 seconds (75 to 115 frames). Naive algorithms trigger a dead ball (`D`) after 1.4 seconds (35 frames).
   - *Solution:* If both the departure position and reappearance position are well within pitch bounds ($x \in [4, 101], y \in [3, 65]$), gaps up to 115 frames (4.6s) are interpolated without triggering false dead balls.
4. **Set-Piece Pre-Kick Hold vs. Kickoff Grace Period:**
   - *Failure:* Treating kickoffs like normal set pieces causes false initial dead balls. Conversely, treating free kicks and corners as live play before the ball is kicked causes false team assignments during dead time.
   - *Solution:* A 50-frame kickoff grace period ensures live play starts immediately at the whistle, while set-piece tagged intervals hold `D` until the ball is actively struck ($v_{\text{ball}} > 3.5\text{ m/s}$).

### 7.2 Results for `submission_v2`:
- **Primary Metric `spellF1`:** Surged to **`0.4667`** (+4.25 points above official baseline).
- **Secondary Metric `macroF1`:** Surged to **`0.7293`** (+14.81 percentage points above baseline).
- **Home Possession Share Error:** Reduced to an astonishing **`0.0012`** (0.12% error across the ground truth).
- **Dead Ball F1:** Reached **`0.6125`** (more than 4x baseline's 0.1464).
- **Execution Speed:** Complete 90-minute evaluation (199,319 frames) executes in **~12.9 seconds** via vectorized NumPy kernels.

---

## 8. Diagnostic Tooling: `starter/inspect_errors.py`

To rapidly pinpoint where points are lost without guessing, we developed `starter/inspect_errors.py`:
- Generates side-by-side segment alignment comparing predicted spells against all 21 GT segments (17 possessions, 4 stoppages).
- Reports exact IoU, duration delta, and error classification (`[EXCELLENT]`, `[GOOD]`, `[WEAK MATCH]`, `[MISSED/SPLIT]`).
- Performs frame-by-frame kinematic inspection at critical timestamps (`00:00` Kickoff, `00:02` Long Ball, `00:11` Pressing Defender, `01:01` 50/50 Aerial Duel).

Run with:
```bash
python starter/inspect_errors.py --pred evaluation/results/submission_v2.csv
```

---

## 9. Next Steps: Roadmap to Version 3 (`v3`)

Analysis from `inspect_errors.py` on `submission_v2` highlights three concrete high-value targets to reach `spellF1 >= 0.50+`:
1. **Contested 50/50 Duel Buffer (`01:00` - `01:04`):**
   - In segment 5/6, an aerial ball drops between two players. Neither player has clean possession for ~1.5s during the header/scramble. Introducing a contested-duel buffer prevents premature state switches before settled control ($v_{\text{ball}} < 4.0\text{ m/s}$ and $d \le 2.0\text{m}$ for $\ge 10$ frames).
2. **Anchor-to-Anchor Bidirectional Bridging:**
   - Define unambiguous "Anchor States" (player in settled control within 1.5m at low speed). If Team X has an anchor at $t_1$, and the next settled anchor at $t_2$ (within 4.0s) is also Team X, bridge the entire interval as Team X regardless of transient proximity to pressing opponents mid-pass.
3. **Camera Homography Integration (`BAR-ATH_tactical_homography.csv`):**
   - Utilize camera homography matrices to verify touchline, endline, and penalty area coordinates to sharpen dead-ball transition boundaries.

