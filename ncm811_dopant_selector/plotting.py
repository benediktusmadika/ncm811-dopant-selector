from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import LabFeedback, SynthesisRow
from .optional_deps import pd, plt, np
from .utils import clamp01
from .constants import LOGGER_NAME

LOG = logging.getLogger(LOGGER_NAME)


def plot_candidate_ranking(ranking: List[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        LOG.warning("matplotlib not installed; skipping plot.")
        return
    if not ranking:
        return
    labels = [str(r.get("dopant_signature", "")) for r in ranking]
    scores = []
    for r in ranking:
        try:
            scores.append(float(r.get("score", 0.0)))
        except Exception:
            scores.append(0.0)
    plt.figure(figsize=(10, max(2.5, 0.45 * len(labels))))
    y = list(range(len(labels)))
    plt.barh(y, scores)
    plt.yticks(y, labels)
    plt.xlabel("Score (0-100)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_subscores_bar(ranking: List[Dict[str, Any]], out_path: Path, top_k: int = 8) -> None:
    """
    Grouped bar chart: evidence_strength vs transferability vs mechanism vs feasibility.
    Scores are assumed to be 0..100.
    """
    if plt is None:
        LOG.warning("matplotlib not installed; skipping subscore bar plot.")
        return
    if not ranking:
        return

    rows = ranking[: max(1, top_k)]
    labels = [str(r.get("dopant_signature", "")) for r in rows]
    cats = [
        ("evidence_strength_score", "evidence"),
        ("transferability_score", "transfer"),
        ("mechanistic_plausibility_score", "mechanism"),
        ("practical_feasibility_score", "feasibility"),
    ]

    # values[c][i]
    vals = []
    for key, _ in cats:
        v = []
        for r in rows:
            try:
                v.append(float(r.get(key, 0.0)))
            except Exception:
                v.append(0.0)
        vals.append(v)

    n = len(labels)
    x = list(range(n))
    width = 0.18
    offsets = [(-1.5 + i) * width for i in range(len(cats))]

    plt.figure(figsize=(11, 3.4 + 0.12 * n))
    for i, (_, lab) in enumerate(cats):
        xs = [xi + offsets[i] for xi in x]
        plt.bar(xs, vals[i], width=width, label=lab)

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Sub-score (0-100)")
    plt.title("Sub-scores per dopant (evidence / transfer / mechanism / feasibility)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_subscores_radar(best_row: Dict[str, Any], out_path: Path) -> None:
    """
    Radar plot for the best dopant (4 axes).
    """
    if plt is None:
        LOG.warning("matplotlib not installed; skipping radar plot.")
        return
    if not best_row:
        return

    cats = [
        ("evidence_strength_score", "evidence"),
        ("transferability_score", "transfer"),
        ("mechanistic_plausibility_score", "mechanism"),
        ("practical_feasibility_score", "feasibility"),
    ]
    vals = []
    for key, _ in cats:
        try:
            vals.append(float(best_row.get(key, 0.0)) / 100.0)
        except Exception:
            vals.append(0.0)

    # close loop
    vals2 = vals + [vals[0]]
    labels = [lab for _, lab in cats]
    labels2 = labels + [labels[0]]

    # angles
    n = len(labels)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles2 = angles + [angles[0]]

    plt.figure(figsize=(5.8, 5.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles2, vals2)
    ax.fill(angles2, vals2, alpha=0.10)
    ax.set_thetagrids([a * 180.0 / math.pi for a in angles], labels)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Radar sub-scores: {best_row.get('dopant_signature','')}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def compute_mismatch_table(
    ranking: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]],
    target: "SynthesisRow",
    top_k: int = 8,
) -> Tuple[List[str], List[str], List[List[float]], List[Dict[str, Any]]]:
    """
    Computes a normalized mismatch matrix in [0,1] for top-k candidates:
      columns = [hydro_T, hydro_time, calc1_T, calc2_T, calc2_time, c_rate, voltage_window]
    Returns (dopant_labels, factor_labels, matrix, row_records).
    """
    by_sig = {str(c.get("dopant_signature") or ""): c for c in (candidate_details or [])}

    # Baseline values
    base_hT = _safe_float(target.hydro_T_C)
    base_hTime = _safe_float(target.hydro_time_h)
    base_c1 = _safe_float(target.calc1_T_C)
    base_c2 = _safe_float(target.calc2_T_C)
    base_c2t = _safe_float(target.calc2_time_h)
    base_cr = _safe_float(target.c_rate)
    base_vw = None  # not present in synthesis.csv typically

    factors = [
        "hydro_T_gap",
        "hydro_time_gap",
        "calc1_T_gap",
        "calc2_T_gap",
        "calc2_time_gap",
        "c_rate_gap",
        "voltage_window_gap",
    ]
    scale = {
        "hydro_T_gap": 50.0,
        "hydro_time_gap": 5.0,
        "calc1_T_gap": 100.0,
        "calc2_T_gap": 100.0,
        "calc2_time_gap": 10.0,
        "c_rate_gap": 1.0,  # log10 ratio
        "voltage_window_gap": 1.0,
    }

    labels: List[str] = []
    mat: List[List[float]] = []
    records: List[Dict[str, Any]] = []

    rows = (ranking or [])[: max(1, top_k)]
    for r in rows:
        sig = str(r.get("dopant_signature") or "").strip()
        det = by_sig.get(sig, {})
        synth = det.get("reported_synthesis_struct") or {}
        cap = det.get("initial_discharge_capacity") or {}

        cand_hT = _safe_float(synth.get("hydro_T_C"))
        cand_hTime = _safe_float(synth.get("hydro_time_h"))
        cand_c1 = _safe_float(synth.get("calc1_T_C"))
        cand_c2 = _safe_float(synth.get("calc2_T_C"))
        cand_c2t = _safe_float(synth.get("calc2_time_h"))
        cand_cr = _safe_float(cap.get("c_rate"))
        cand_vw = cap.get("voltage_window")

        def gap_abs_diff(cand: Optional[float], base: Optional[float], s: float) -> float:
            if cand is None or base is None:
                return float("nan")
            return min(1.0, abs(cand - base) / max(1e-9, s))

        def gap_c_rate(cand: Optional[float], base: Optional[float]) -> float:
            if cand is None or base is None or base <= 0 or cand <= 0:
                return float("nan")
            return min(1.0, abs(math.log10(cand / base)) / 1.0)

        def gap_voltage(cand: Any, base: Any) -> float:
            if cand is None or base is None:
                return float("nan")
            cs = str(cand).replace(" ", "")
            bs = str(base).replace(" ", "")
            return 0.0 if cs == bs else 1.0

        row = [
            gap_abs_diff(cand_hT, base_hT, scale["hydro_T_gap"]),
            gap_abs_diff(cand_hTime, base_hTime, scale["hydro_time_gap"]),
            gap_abs_diff(cand_c1, base_c1, scale["calc1_T_gap"]),
            gap_abs_diff(cand_c2, base_c2, scale["calc2_T_gap"]),
            gap_abs_diff(cand_c2t, base_c2t, scale["calc2_time_gap"]),
            gap_c_rate(cand_cr, base_cr),
            gap_voltage(cand_vw, base_vw),
        ]
        labels.append(sig)
        mat.append(row)

        records.append({
            "dopant_signature": sig,
            "baseline_hydro_T_C": base_hT,
            "candidate_hydro_T_C": cand_hT,
            "baseline_hydro_time_h": base_hTime,
            "candidate_hydro_time_h": cand_hTime,
            "baseline_calc1_T_C": base_c1,
            "candidate_calc1_T_C": cand_c1,
            "baseline_calc2_T_C": base_c2,
            "candidate_calc2_T_C": cand_c2,
            "baseline_calc2_time_h": base_c2t,
            "candidate_calc2_time_h": cand_c2t,
            "baseline_c_rate": base_cr,
            "candidate_c_rate": cand_cr,
            "baseline_voltage_window": base_vw,
            "candidate_voltage_window": cand_vw,
        })

    return labels, factors, mat, records

def plot_mismatch_heatmap(
    ranking: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]],
    target: "SynthesisRow",
    out_path: Path,
    top_k: int = 8,
    export_csv: Optional[Path] = None,
) -> None:
    """
    Heatmap of normalized mismatch (0=match, 1=large mismatch; NaN=unknown).
    """
    if plt is None:
        LOG.warning("matplotlib not installed; skipping mismatch heatmap.")
        return
    labels, factors, mat, records = compute_mismatch_table(ranking, candidate_details, target, top_k=top_k)
    if not labels:
        return

    # Export raw mismatch values for paper reproducibility
    if export_csv is not None:
        try:
            export_csv.parent.mkdir(parents=True, exist_ok=True)
            if pd is not None:
                pd.DataFrame(records).to_csv(export_csv, index=False)
            else:
                import csv
                with open(export_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["dopant_signature"])
                    w.writeheader()
                    for r in records:
                        w.writerow(r)
        except Exception as e:
            LOG.warning("Failed to export mismatch CSV: %s", e)

    # Convert to numpy for masked NaNs if available
    if np is not None:
        arr = np.array(mat, dtype=float)
        arr_masked = np.ma.masked_invalid(arr)
        data = arr_masked
    else:
        data = mat

    plt.figure(figsize=(11, 0.45 * len(labels) + 1.6))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Normalized mismatch (0=match, 1=large gap)")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(factors)), factors, rotation=25, ha="right")
    plt.title("Mismatch heatmap: candidate vs baseline synthesis/test parameters")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_mechanism_map_heatmap(mech_map: Dict[str, Any], out_path: Path) -> None:
    """
    Heatmap: dopant x mechanism category.
    Values are signed strengths: +strength (positive), -strength (negative), 0 (unclear).
    """
    if plt is None:
        LOG.warning("matplotlib not installed; skipping mechanism map heatmap.")
        return
    if not mech_map or not isinstance(mech_map, dict):
        return

    categories = mech_map.get("categories") or MECH_CATEGORIES
    dopants = mech_map.get("dopants") or []
    if not categories or not dopants:
        return

    labels = []
    mat = []
    for d in dopants:
        sig = str(d.get("dopant_signature") or "").strip()
        if not sig:
            continue
        labels.append(sig)
        row = [0.0 for _ in categories]
        links = d.get("category_links") or []
        for lk in links:
            cat = str(lk.get("category") or "").strip()
            if cat not in categories:
                continue
            j = categories.index(cat)
            direction = str(lk.get("direction") or "unclear").lower()
            strength = _safe_float(lk.get("strength")) or 0.0
            strength = clamp01(float(strength))
            if direction.startswith("pos"):
                row[j] = max(row[j], +strength)
            elif direction.startswith("neg"):
                row[j] = min(row[j], -strength)
            else:
                row[j] = row[j]  # keep
        mat.append(row)

    if not labels:
        return

    if np is not None:
        arr = np.array(mat, dtype=float)
    else:
        arr = mat

    plt.figure(figsize=(12, 0.45 * len(labels) + 1.6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="Signed mechanism strength (+ helps, - hurts initial capacity)")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(categories)), categories, rotation=25, ha="right")
    plt.title("Mechanism map (GPT-assisted): dopant → mechanism categories → direction on initial capacity")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_closed_loop_trajectory(trials: List["LabFeedback"], out_path: Path, baseline_capacity: Optional[float] = None) -> None:
    """
    Plot iteration vs measured initial/first discharge capacity, annotated by dopant/level.
    """
    if plt is None:
        LOG.warning("matplotlib not installed; skipping closed-loop trajectory plot.")
        return
    if not trials:
        return

    xs = [int(t.trial_id) for t in trials]
    ys = [float(t.measured_initial_discharge_mAh_g) for t in trials]
    ann = []
    for t in trials:
        f = "" if t.doping_fraction is None else f"{float(t.doping_fraction):.3f}"
        ann.append(f"{t.dopant_signature} {f}".strip())

    plt.figure(figsize=(9.5, 3.6))
    plt.plot(xs, ys, marker="o")
    if baseline_capacity is not None:
        try:
            plt.axhline(float(baseline_capacity), linestyle="--")
        except Exception:
            pass
    for x, y, a in zip(xs, ys, ann):
        plt.text(x, y, " " + a, fontsize=8, va="center")

    plt.xlabel("Iteration / trial_id")
    plt.ylabel("Measured initial (first) discharge capacity (mAh/g)")
    plt.title("Closed-loop trajectory (lab feedback)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def export_long_csv(candidate_details: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Export a long-format CSV of candidate evidence.
    """
    rows: List[Dict[str, Any]] = []
    for c in candidate_details:
        cap = c.get("initial_discharge_capacity") or {}
        rows.append({
            "dopant_signature": c.get("dopant_signature"),
            "modifier_mode": c.get("modifier_mode"),
            "doping_level_text": c.get("doping_level_text"),
            "doping_fraction": c.get("doping_fraction"),
            "doping_basis": c.get("doping_basis"),
            "dopant_precursors": "|".join(c.get("dopant_precursors") or []),
            "trend": cap.get("trend"),
            "doped_value_mAh_g": cap.get("doped_value_mAh_g"),
            "baseline_value_mAh_g": cap.get("baseline_value_mAh_g"),
            "delta_mAh_g": cap.get("delta_mAh_g"),
            "c_rate": cap.get("c_rate"),
            "voltage_window": cap.get("voltage_window"),
            "evidence_chunks": "|".join(cap.get("evidence_chunks") or []),
            "quotes": " || ".join((cap.get("quotes") or [])[:3]),
            "confidence": cap.get("confidence"),
            "overall_confidence": c.get("overall_confidence"),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dopant_signature"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
