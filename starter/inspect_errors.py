"""Diagnostic tool to inspect errors between submission predictions and Ground Truth.

Compares predictions against evaluation/gt/BAR-ATH_possession_gt_public.csv across the 7,500
graded public frames (11185..18684). Produces:
  1. A structured side-by-side comparison of all 17 GT possessions & 4 stoppages vs predicted spells.
  2. Classification of specific error modes (Spurious D, Broken Pass, Transition Lag, Opponent False Trigger).
  3. Frame-level deep dives into critical video timestamps (00:00 Kickoff, 00:02 Long Ball, 00:11 Short Pass, 01:01 Aerial Duel).

Usage:
    python starter/inspect_errors.py --pred evaluation/results/submission_v1.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from games import GAME, GAMES, require

PUBLIC_GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_PRED_PATH = os.path.join(ROOT, "evaluation", "results", "submission_v1.csv")

LABEL_NAMES = {pm.H: "H (Barcelona)", pm.A: "A (Athletic)", pm.D: "D (Dead Ball)", pm.X: "X"}


def format_time(video_frame, kickoff_frame=11185, fps=25):
    """Converts a video frame into match clock (MM:SS.s)."""
    rel_frames = max(0, video_frame - kickoff_frame)
    total_seconds = rel_frames / float(fps)
    mins = int(total_seconds // 60)
    secs = total_seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def inspect_comparison(pred_path=DEFAULT_PRED_PATH, gt_path=PUBLIC_GT_PATH):
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    gt_df = pm.read_label_csv(gt_path)
    pred_df = pm.read_label_csv(pred_path)

    gt_codes = pm.label_codes(gt_df)
    pred_codes = pm.label_codes(pred_df)
    runs = pm.valid_runs(gt_codes)

    if not runs:
        print("No valid graded runs found in Ground Truth.")
        return

    v_start, v_stop = runs[0]
    gt_slice = gt_codes[v_start:v_stop]
    pred_slice = pred_codes[v_start:v_stop]

    gt_segs = pm.segments(gt_codes, runs)
    pred_segs = pm.segments(pred_codes, runs)

    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(pred_codes, runs)
    scores = pm.spell_scores(gt_codes, pred_codes, runs)
    eval_metrics = pm.evaluate(gt_codes, pred_codes)

    print("=" * 110)
    print(f"DIAGNOSTIC REPORT: {os.path.basename(pred_path)} vs PUBLIC GROUND TRUTH")
    print(f"Scored Frames: {v_start}..{v_stop-1} ({v_stop - v_start} frames, ~5 minutes)")
    trans_f1_loose = eval_metrics.get("transitions", {}).get(25, {}).get("f1", 0.0)
    print(f"Headline Metrics: spellF1 = {scores['mean_f1']:.4f} | macroF1 = {eval_metrics['macro_f1']:.4f} | transF1(+-25) = {trans_f1_loose:.4f}")
    print(f"Possessions: GT = {len(gt_spells)} | Predicted = {len(pred_spells)} | Fragmentation = {eval_metrics['fragmentation']:.2f}")
    print("=" * 110)

    print("\n--- SIDE-BY-SIDE SEGMENT ALIGNMENT (All 21 GT Segments) ---")
    print(f"{'#':<3} {'Time Range':<15} {'Frames':<15} {'GT':<4} {'Dur':<6} | {'Predicted Overlapping Segments & IoU':<55}")
    print("-" * 110)

    for i, (g_s, g_e, g_c) in enumerate(gt_segs, 1):
        g_team = "H" if g_c == pm.H else ("A" if g_c == pm.A else "D")
        dur_s = (g_e - g_s) / 25.0
        time_str = f"{format_time(g_s)} - {format_time(g_e)}"
        frame_str = f"{g_s}..{g_e-1}"

        # Overlapping predicted segments
        overlaps = []
        for p_s, p_e, p_c in pred_segs:
            if p_e > g_s and p_s < g_e:
                p_team = "H" if p_c == pm.H else ("A" if p_c == pm.A else "D")
                inter = min(g_e, p_e) - max(g_s, p_s)
                union = max(g_e, p_e) - min(g_s, p_s)
                iou = inter / float(union) if p_c == g_c else 0.0
                p_dur = (p_e - p_s) / 25.0
                overlaps.append(f"{p_team}[{p_s}..{p_e-1}, {p_dur:.1f}s, IoU={iou:.2f}]")

        overlap_str = " | ".join(overlaps) if overlaps else "NONE"
        status = ""
        if g_team in ["H", "A"]:
            # Check spell IoU
            matched_iou = [float(x.split("IoU=")[-1].replace("]", "")) for x in overlaps if g_team in x]
            max_iou = max(matched_iou) if matched_iou else 0.0
            if max_iou >= 0.7:
                status = "EXCELLENT"
            elif max_iou >= 0.5:
                status = "GOOD"
            elif max_iou >= 0.3:
                status = "WEAK MATCH"
            else:
                status = "MISSED/SPLIT"
        else:
            status = "DEAD BALL"

        print(f"{i:<3} {time_str:<15} {frame_str:<15} {g_team:<4} {dur_s:4.1f}s | {overlap_str:<55} [{status}]")

    print("\n" + "=" * 110)
    print("KEY VIDEO TIMESTAMPS: KINEMATIC DEEP DIVE")
    print("=" * 110)

    # Detailed inspection of specific user moments:
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    length, width = GAMES[GAME]["pitch"]

    deep_dives = [
        ("00:00 - 00:05 (Kickoff & Aerial Long Ball)", 11185, 11260),
        ("00:11 (Short Pass with Opponent Nearby)", 11450, 11490),
        ("01:00 - 01:04 (Long Ball, 50/50 Aerial Duel & Second Ball)", 12675, 12745),
    ]

    for title, f_start, f_end in deep_dives:
        print(f"\n>>> Timestamp: {title} (Frames {f_start}..{f_end})")
        print(f"{'Frame':<7} {'Clock':<8} {'GT':<4} {'v1':<4} {'Match?':<7} {'Ball (x, y)':<18} {'Ball Spd':<10} {'Home d':<8} {'Away d':<8} {'Kinematic Context'}")
        print("-" * 110)

        # We read raw tracking for these frames
        frames_info = {}
        with open(raw_path) as h:
            for line in h:
                parts = line.split(":")
                if len(parts) != 3: continue
                fid = int(parts[0]) - 1
                if f_start <= fid <= f_end:
                    ball = parts[2].strip()
                    bx, by, bspd = np.nan, np.nan, np.nan
                    if not ball.startswith("N"):
                        bparts = ball.split(",")
                        bx = float(bparts[0]) * length
                        by = float(bparts[1]) * width
                    frames_info[fid] = (bx, by)

        prev_bx, prev_by = None, None
        for f in range(f_start, f_end + 1):
            gt_lbl = pm.LABELS[gt_codes[f]]
            v1_lbl = pm.LABELS[pred_codes[f]]
            bx, by = frames_info.get(f, (np.nan, np.nan))
            spd_str = "MISSING"
            if not np.isnan(bx):
                if prev_bx is not None and not np.isnan(prev_bx):
                    spd = (((bx - prev_bx)**2 + (by - prev_by)**2)**0.5) * 25.0
                    spd_str = f"{spd:5.1f} m/s"
                else:
                    spd_str = "  N/A   "
                prev_bx, prev_by = bx, by
            else:
                prev_bx, prev_by = None, None

            ball_pos = f"({bx:5.1f}, {by:5.1f})" if not np.isnan(bx) else "  MISSING BALL  "
            match_status = "OK" if gt_lbl == v1_lbl else "MISMATCH"

            # Context notes
            note = ""
            if f == 11185: note = "Kickoff taken by Athletic (A)"
            elif 11200 <= f <= 11215: note = "Defender launching long ball"
            elif 11216 <= f <= 11245: note = "Ball high in sunlight (tracking loss risk)"
            elif 11460 <= f <= 11475: note = "Short pass: Barcelona passing, Athletic pressing"
            elif 11483 == f: note = "v1 switched from A to H here"
            elif 12680 <= f <= 12700: note = "Long ball drops: 50/50 aerial duel contest"
            elif 12705 <= f <= 12720: note = "Second ball scramble before settled control"

            clock_str = format_time(f)
            # Sample every 2-3 frames if too many, or print continuous during transitions
            if (f_end - f_start <= 45) or (f % 3 == 0) or (gt_lbl != v1_lbl):
                print(f"{f:<7} {clock_str:<8} {gt_lbl:<4} {v1_lbl:<4} {match_status:<7} {ball_pos:<18} {spd_str:<10} {note}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", default=DEFAULT_PRED_PATH, help="Path to prediction CSV")
    parser.add_argument("--gt", default=PUBLIC_GT_PATH, help="Path to Ground Truth CSV")
    args = parser.parse_args()

    inspect_comparison(args.pred, args.gt)
