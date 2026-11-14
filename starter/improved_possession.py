"""Improved possession inference combining ATD tracking, ball kinematics, Smart Tagging, and hysteresis.

Supports:
  - Version 1 (v1): Baseline extension with linear ball gap interpolation, basic persistence state machine,
    tactical phase priors, and minimum spell duration filtering.
  - Version 2 (v2 - Kinematic Upgrade):
    1. Directional Velocity (Cosine Alignment): Evaluates the dot product between the ball's movement
       vector and the player vector. Fast-moving balls traveling away from pressing opponents incur an
       effective distance penalty, eliminating false turnover triggers on passes.
    2. Velocity-Gated Possession Control: Ball speeds exceeding 11.5 m/s cannot be stolen simply by proximity
       as a long ball or clearance flies past a defender.
    3. In-Bounds Aerial Occlusion Handling: Bridges long-ball trajectories where the ball vanishes into sunlight
       or high camera angles (up to 115 frames / 4.6s) as long as departure and arrival are well inside pitch limits.
    4. Set-Piece Pre-Kick Dead-Time Anchors: Recognizes that during set pieces (Goal Kicks, Free Kicks, Corners),
       play remains Dead (D) until the ball is actively struck and launched into play.
    5. Hysteresis Filtering: Eliminates micro-flickers under 2.6s (65 frames) to mirror natural physical possessions.

Usage:
    python starter/improved_possession.py --version v2 --out evaluation/results/submission_v2.csv
    python evaluation/validate_submission.py evaluation/results/submission_v2.csv
    python evaluation/run_evaluation.py
"""

import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
import starter.load_smart_tagging as lst  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, -1
LABELS = np.array(["H", "A", "D", "X"])

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"


def team_of_each_track(meta_path):
    """Array over track slots: 0 for home team, 1 for away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def extract_full_tracking(raw_path, n_frames, pitch):
    """Fast streaming parse of ball coordinates and all 46 player slot positions."""
    length, width = pitch
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    player_x = np.full((n_frames, 46), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, 46), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid < 0 or fid >= n_frames:
                continue
            ball = parts[2]
            if ball[0] != MISSING:
                comma = ball.index(",")
                ball_x[fid] = float(ball[:comma]) * length
                ball_y[fid] = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    comma = group.index(",")
                    player_x[fid, slot] = float(group[:comma]) * length
                    player_y[fid, slot] = float(group[comma + 1:group.index(",", comma + 1)]) * width

    return ball_x, ball_y, player_x, player_y


def interpolate_inbounds_aerial_gaps(ball_x, ball_y, length, width, max_aerial_gap=115, max_ground_gap=35):
    """Interpolate ball trajectories, allowing longer gaps for in-bounds aerial flights."""
    is_missing = np.isnan(ball_x)
    diffs = np.diff(np.concatenate(([False], is_missing, [False])).astype(int))
    gap_starts = np.where(diffs == 1)[0]
    gap_ends = np.where(diffs == -1)[0]
    n_frames = len(ball_x)

    for gs, ge in zip(gap_starts, gap_ends):
        gap_len = ge - gs
        if gs > 0 and ge < n_frames:
            bx_prev, by_prev = ball_x[gs - 1], ball_y[gs - 1]
            bx_next, by_next = ball_x[ge], ball_y[ge]

            # In-bounds check: departure and arrival both inside pitch boundaries
            in_bounds_prev = (4.0 <= bx_prev <= length - 4.0) and (3.0 <= by_prev <= width - 3.0)
            in_bounds_next = (4.0 <= bx_next <= length - 4.0) and (3.0 <= by_next <= width - 3.0)

            max_allowed = max_aerial_gap if (in_bounds_prev and in_bounds_next) else max_ground_gap
            if gap_len <= max_allowed:
                t_span = np.arange(gap_len + 2)
                ball_x[gs:ge] = np.interp(t_span, [0, gap_len + 1], [bx_prev, bx_next])[1:-1]
                ball_y[gs:ge] = np.interp(t_span, [0, gap_len + 1], [by_prev, by_next])[1:-1]


def compute_directional_effective_distances(ball_x, ball_y, player_x, player_y, team_of_track):
    """Compute ball velocity vectors and directional cosine-aligned effective distances for both teams."""
    n_frames = len(ball_x)
    ball_vx = np.zeros(n_frames, dtype=np.float32)
    ball_vy = np.zeros(n_frames, dtype=np.float32)
    ball_vx[1:] = (ball_x[1:] - ball_x[:-1]) * 25.0
    ball_vy[1:] = (ball_y[1:] - ball_y[:-1]) * 25.0
    ball_spd = np.sqrt(ball_vx**2 + ball_vy**2)

    dx = player_x - ball_x[:, None]
    dy = player_y - ball_y[:, None]
    dist = np.sqrt(dx**2 + dy**2)

    # Cosine similarity between ball velocity and ball-to-player vector
    dot = dx * ball_vx[:, None] + dy * ball_vy[:, None]
    cos_sim = dot / (dist * ball_spd[:, None] + 1e-6)

    dist_eff = dist.copy()
    fast_mask = (ball_spd[:, None] > 5.0) & (~np.isnan(dist))

    # Penalty for opponents/players moving away from ball
    away_pen_mask = fast_mask & (cos_sim < -0.2)
    dist_eff[away_pen_mask] += np.minimum(3.0, ball_spd[:, None] * 0.25 * (-cos_sim))[away_pen_mask]

    # Bonus for players directly in the path of the ball (receivers)
    toward_bon_mask = fast_mask & (cos_sim > 0.5)
    dist_eff[toward_bon_mask] = np.maximum(0.2, dist_eff - np.minimum(0.8, ball_spd[:, None] * 0.1 * cos_sim))[toward_bon_mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        home_eff = np.nanmin(dist_eff[:, team_of_track == HOME], axis=1)
        away_eff = np.nanmin(dist_eff[:, team_of_track == AWAY], axis=1)

    # Fallback for any NaN frames
    home_eff = np.where(np.isnan(home_eff), 999.0, home_eff)
    away_eff = np.where(np.isnan(away_eff), 999.0, away_eff)

    return ball_spd, home_eff, away_eff


def filter_short_spells(codes, min_spell_len=65):
    """Eliminate transient spells shorter than min_spell_len frames via hysteresis merging."""
    if min_spell_len <= 1:
        return codes
    out = codes.copy()
    breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(out)]))

    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        length = e - s
        if length < min_spell_len:
            left = out[s - 1] if s > 0 else None
            right = out[e] if e < len(out) else None
            if left is not None and right is not None and left == right:
                out[s:e] = left
            elif left is not None:
                out[s:e] = left
            elif right is not None:
                out[s:e] = right
    return out


def build_submission(version="v2", game=GAME, data_dir=None, min_spell_len=65, control_dist=3.5, duel_margin=1.0, switch_frames=10):
    """Run possession inference over the full match."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    length, width = pitch
    team_of_track = team_of_each_track(meta_path)
    home_name, away_name = team_names(meta_path)

    print(f"Reading ATD tracking from {os.path.basename(raw_path)} ({n_frames} frames)...")
    t0 = time.time()
    ball_x, ball_y, player_x, player_y = extract_full_tracking(raw_path, n_frames, pitch)
    print(f"  Extracted coordinates in {time.time() - t0:.2f}s.")

    # In-bounds aerial occlusion interpolation
    print("Interpolating in-bounds aerial trajectories and short gaps...")
    interpolate_inbounds_aerial_gaps(ball_x, ball_y, length, width, max_aerial_gap=115, max_ground_gap=35)

    # Compute directional effective distances
    print("Computing kinematics: ball velocity vector & directional cosine proximity...")
    ball_spd, home_eff, away_eff = compute_directional_effective_distances(
        ball_x, ball_y, player_x, player_y, team_of_track
    )

    # Load Smart Tagging tactical annotations
    print("Loading Smart Tagging tactical phases and set pieces...")
    events = lst.load(game=game, data_dir=data_dir)
    tag_team_phase = np.full(n_frames, -1, dtype=np.int8)
    set_pieces = []

    for ev in events:
        sf, ef = ev['start_frame'], ev['end_frame']
        code = ev['code'].upper()
        team = ev['tags'].get('Team', '')
        p_type = ev['tags'].get('Type', '')
        t_code = HOME if team == home_name else (AWAY if team == away_name or 'ATLETIC' in team.upper() else -1)

        s = max(0, sf)
        e = min(n_frames, ef)
        if s >= e: continue

        if code == 'SET PIECES':
            # Exclude kickoff from dead ball (kickoffs start play immediately)
            if 'KICK OFF' not in p_type.upper():
                set_pieces.append((s, e, t_code, p_type))
        elif code in ['PROGRESSION', 'BUILD UP', 'BALL IN FINAL THIRD', 'BALL IN THE BOX', 'CREATING CHANCES']:
            if t_code != -1:
                tag_team_phase[s:e] = t_code

    print(f"  Loaded {len(set_pieces)} set piece stoppage intervals.")

    # Process graded half blocks
    halves = half_video_blocks(game)
    full_labels = np.full(n_frames, "X", dtype="<U1")

    for half_num, h_start, h_end in halves:
        print(f"Processing Half {half_num}: frames {h_start}..{h_end} ({h_end - h_start + 1} frames)...")
        sub_len = h_end - h_start + 1

        sub_bx = ball_x[h_start:h_end + 1]
        sub_by = ball_y[h_start:h_end + 1]
        sub_bspd = ball_spd[h_start:h_end + 1]
        sub_hd = home_eff[h_start:h_end + 1]
        sub_ad = away_eff[h_start:h_end + 1]
        sub_phase = tag_team_phase[h_start:h_end + 1]

        # Initial possessor at kickoff: Athletic Club kicked off 1st half, Barcelona 2nd half
        current_state = AWAY if half_num == 1 else HOME
        consec_h = 0
        consec_a = 0

        pred_codes = np.full(sub_len, DEAD, dtype=np.int8)

        # Slice relevant set pieces
        half_set_pieces = [(max(0, s - h_start), min(sub_len, e - h_start), t, p)
                           for s, e, t, p in set_pieces if e >= h_start and s <= h_end]

        for i in range(sub_len):
            bx = sub_bx[i]
            by = sub_by[i]
            b_s = sub_bspd[i]
            hd = sub_hd[i]
            ad = sub_ad[i]

            # 1. Missing ball after in-bounds aerial interpolation -> DEAD
            if np.isnan(bx):
                # Kickoff grace period (first 50 frames of half): stays kickoff team
                if i < 50:
                    pred_codes[i] = current_state
                    continue
                current_state = DEAD
                consec_h = consec_a = 0
                pred_codes[i] = DEAD
                continue

            # 2. Out of bounds -> DEAD
            if bx < -0.3 or bx > length + 0.3 or by < -0.3 or by > width + 0.3:
                current_state = DEAD
                consec_h = consec_a = 0
                pred_codes[i] = DEAD
                continue

            # 3. Kinematic v2: Set Piece pre-kick dead time
            if version == "v2":
                sp_match = [sp for sp in half_set_pieces if sp[0] <= i <= sp[1]]
                if sp_match:
                    s_s, s_e, s_team, s_type = sp_match[0]
                    # While ball is being placed or held before restart kick (ball speed < 3.5 m/s)
                    if b_s < 3.5 and (i < s_e - 15):
                        current_state = DEAD
                        consec_h = consec_a = 0
                        pred_codes[i] = DEAD
                        continue

            min_p = min(hd, ad)
            if current_state == DEAD:
                # Controlled restart condition: ball is back in play and controlled
                if min_p <= 2.5:
                    current_state = HOME if hd < ad else AWAY
                pred_codes[i] = current_state
                continue

            # 4. Kinematic v2: Velocity-Gated Possession Control
            # If ball is traveling at high speed (> 11.5 m/s), an opponent cannot steal possession mid-flight
            if version == "v2" and b_s > 11.5:
                pred_codes[i] = current_state
                continue

            # 5. Live play state machine with directional effective distance & tactical prior
            bias_h = 0.2 if sub_phase[i] == HOME else 0.0
            bias_a = 0.2 if sub_phase[i] == AWAY else 0.0
            eff_hd = hd - bias_h
            eff_ad = ad - bias_a

            if eff_hd < eff_ad - duel_margin and hd < control_dist:
                consec_h += 1
                consec_a = 0
            elif eff_ad < eff_hd - duel_margin and ad < control_dist:
                consec_a += 1
                consec_h = 0
            else:
                consec_h = max(0, consec_h - 1)
                consec_a = max(0, consec_a - 1)

            if current_state == HOME:
                if consec_a >= switch_frames:
                    current_state = AWAY
                    consec_a = 0
            elif current_state == AWAY:
                if consec_h >= switch_frames:
                    current_state = HOME
                    consec_h = 0

            pred_codes[i] = current_state

        # Apply minimum spell duration filtering
        pred_codes = filter_short_spells(pred_codes, min_spell_len=min_spell_len)

        char_labels = np.where(pred_codes == HOME, "H", np.where(pred_codes == AWAY, "A", "D"))
        full_labels[h_start:h_end + 1] = char_labels

    return full_labels


def write_submission(path, labels):
    """Write submission CSV in exact official format."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=["v1", "v2"], default="v2",
                        help="algorithm version (v1: baseline extension, v2: kinematics upgrade)")
    parser.add_argument("--out", default=None,
                        help="where to write the submission CSV (default: evaluation/results/submission_<version>.csv)")
    parser.add_argument("--min-spell", type=int, default=65,
                        help="minimum spell duration in frames (default: 65 = 2.6s)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(RESULTS_DIR, f"submission_{args.version}.csv")

    t_start = time.time()
    labels = build_submission(version=args.version, min_spell_len=args.min_spell)
    path = write_submission(out_path, labels)

    counts = {l: int(np.sum(labels == l)) for l in LABELS}
    print(f"\nWrote {os.path.relpath(path, ROOT)} in {time.time() - t_start:.2f}s")
    print(f"Summary ({args.version}): H={counts['H']}  A={counts['A']}  D={counts['D']}  X={counts['X']}")
    print(f"Total rows: {len(labels)}")
