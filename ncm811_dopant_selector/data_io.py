from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LabFeedback, SynthesisRow
from .optional_deps import pd


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def load_synthesis_csv(path: Path) -> List[SynthesisRow]:
    if not path.exists():
        raise FileNotFoundError(path)
    if pd is None:
        raise RuntimeError("pandas is required for synthesis.csv parsing. Install pandas.")

    # robust encodings
    df = None
    for enc in ("utf-8", "utf-8-sig", "ISO-8859-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(path, encoding_errors="replace")

    rows: List[SynthesisRow] = []
    for i, r in enumerate(df.to_dict(orient="records")):
        # determine method
        method = "unknown"
        htT = _try_float(r.get("Hydrothermal_Temperature"))
        htH = _try_float(r.get("Hydrothermal_Time"))
        water = str(r.get("Water_Solvent", "")).strip().lower()
        add = str(r.get("Additional_Solvent", "")).strip().lower()
        if htT is not None or htH is not None:
            if add in {"yes","y","true","1"} and water in {"yes","y","true","1"}:
                method = "solvothermal"
            else:
                method = "hydrothermal"

        # calc times: your CSV has inconsistent column names; handle gracefully
        calc1_T = _try_float(r.get("1st_Calcination_Temperature"))
        # Some datasets mistakenly store first calcination time in a "2nd_Calcination_Time" column (like your sample)
        calc1_time = _try_float(r.get("1st_Calcination_Time")) or _try_float(r.get("2nd_Calcination_Time"))
        calc2_T = _try_float(r.get("2nd_Calcination_ Temperature")) or _try_float(r.get("2nd_Calcination_Temperature"))
        calc2_time = _try_float(r.get("2nd_Calcination_ Time")) or _try_float(r.get("2nd_Calcination_Time"))

        rows.append(SynthesisRow(
            row_id=i,
            method=method,
            li_precursor=str(r.get("Li_Precursor","")).strip(),
            ni_precursor=str(r.get("Ni_Precursor","")).strip(),
            co_precursor=str(r.get("Co_Precursor","")).strip(),
            mn_precursor=str(r.get("Mn_Precursor","")).strip(),
            hydro_T_C=htT,
            hydro_time_h=htH,
            calc1_T_C=calc1_T,
            calc1_time_h=calc1_time,
            calc2_T_C=calc2_T,
            calc2_time_h=calc2_time,
            c_rate=_try_float(r.get("C_Rate")),
            baseline_capacity_mAh_g=_try_float(r.get("Discharge_ Capacity (mAh/g)")),
            raw={k: r[k] for k in r},
        ))
    return rows

def _jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip corrupted line
                continue
    return out

def _jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_feedback_jsonl(path: Path, synthesis_row_id: Optional[int] = None) -> List[LabFeedback]:
    raws = _jsonl_read(path)
    trials: List[LabFeedback] = []
    for r in raws:
        try:
            if synthesis_row_id is not None and int(r.get("synthesis_row_id", -1)) != int(synthesis_row_id):
                continue
            trials.append(LabFeedback(
                trial_id=int(r.get("trial_id", len(trials))),
                timestamp=str(r.get("timestamp", "")),
                synthesis_row_id=int(r.get("synthesis_row_id", -1)),
                dopant_signature=str(r.get("dopant_signature", "")),
                doping_fraction=_try_float(r.get("doping_fraction")),
                doping_basis=str(r.get("doping_basis", "unknown")),
                modifier_mode=str(r.get("modifier_mode", "unknown")),
                dopant_precursors=list(r.get("dopant_precursors") or []),
                measured_initial_discharge_mAh_g=float(r.get("measured_initial_discharge_mAh_g")),
                measured_c_rate=float(r.get("measured_c_rate", 0.1)),
                voltage_window=(None if r.get("voltage_window") in (None, "", "null") else str(r.get("voltage_window"))),
                notes=(None if r.get("notes") in (None, "", "null") else str(r.get("notes"))),
                source_recipe_path=(None if r.get("source_recipe_path") in (None, "", "null") else str(r.get("source_recipe_path"))),
            ))
        except Exception:
            continue
    # sort by trial_id
    trials.sort(key=lambda t: int(t.trial_id))
    return trials

def summarize_feedback_for_prompt(
    trials: List[LabFeedback],
    baseline_capacity_mAh_g: Optional[float],
    max_trials: int = 12,
) -> Dict[str, Any]:
    """
    Keep feedback context compact:
    - Always include the most recent trials (up to 8)
    - Always include best and worst trial
    """
    if not trials:
        return {"baseline_capacity_mAh_g": baseline_capacity_mAh_g, "trials": [], "summary": "no_feedback"}

    # Sort by time (trial_id proxy) and by performance
    trials_by_time = sorted(trials, key=lambda t: int(t.trial_id))
    trials_by_perf = sorted(trials, key=lambda t: float(t.measured_initial_discharge_mAh_g), reverse=True)

    keep: List[LabFeedback] = []
    keep.extend(trials_by_time[-8:])
    keep.append(trials_by_perf[0])
    keep.append(trials_by_perf[-1])

    # unique preserve order by recipe_key + trial_id
    seen = set()
    uniq: List[LabFeedback] = []
    for t in keep:
        key = (t.recipe_key(), int(t.trial_id))
        if key not in seen:
            seen.add(key)
            uniq.append(t)

    # cap
    uniq = uniq[:max_trials]

    # build compact dicts
    trial_dicts: List[Dict[str, Any]] = []
    for t in uniq:
        delta = None
        if baseline_capacity_mAh_g is not None:
            delta = float(t.measured_initial_discharge_mAh_g) - float(baseline_capacity_mAh_g)
        trial_dicts.append({
            "trial_id": t.trial_id,
            "dopant_signature": t.dopant_signature,
            "doping_fraction": t.doping_fraction,
            "doping_basis": t.doping_basis,
            "modifier_mode": t.modifier_mode,
            "dopant_precursors": t.dopant_precursors,
            "measured_initial_discharge_mAh_g": t.measured_initial_discharge_mAh_g,
            "measured_c_rate": t.measured_c_rate,
            "voltage_window": t.voltage_window,
            "delta_vs_baseline_mAh_g": delta,
            "notes": t.notes,
        })

    best = trials_by_perf[0]
    best_delta = None if baseline_capacity_mAh_g is None else float(best.measured_initial_discharge_mAh_g) - float(baseline_capacity_mAh_g)
    summary = f"{len(trials)} trial(s). Best so far: {best.dopant_signature} @ {best.doping_fraction} ({best.measured_initial_discharge_mAh_g} mAh/g, Δ={best_delta})."

    return {
        "baseline_capacity_mAh_g": baseline_capacity_mAh_g,
        "trials": trial_dicts,
        "summary": summary,
    }

def load_recipe_minimal(recipe_path: Path) -> Dict[str, Any]:
    data = json.loads(recipe_path.read_text(encoding="utf-8"))
    # Supports best_recipe.json from this pipeline.
    return {
        "dopant_signature": str(data.get("dopant_signature", "")),
        "dopant_elements": data.get("dopant_elements") or [],
        "doping_fraction": data.get("doping_fraction"),
        "doping_basis": str(data.get("doping_basis", "unknown")),
        "modifier_mode": str((data.get("best_detail") or data).get("modifier_mode", data.get("modifier_mode", "unknown"))),
        "dopant_precursors": data.get("dopant_precursors") or [],
    }

def build_feedback_record(
    *,
    target: SynthesisRow,
    trial_id: int,
    recipe: Dict[str, Any],
    measured_capacity_mAh_g: float,
    measured_c_rate: float,
    voltage_window: Optional[str],
    notes: Optional[str],
    recipe_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "trial_id": int(trial_id),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthesis_row_id": int(target.row_id),
        "dopant_signature": str(recipe.get("dopant_signature", "")).strip(),
        "doping_fraction": recipe.get("doping_fraction", None),
        "doping_basis": str(recipe.get("doping_basis", "unknown")),
        "modifier_mode": str(recipe.get("modifier_mode", "unknown")),
        "dopant_precursors": list(recipe.get("dopant_precursors") or []),
        "measured_initial_discharge_mAh_g": float(measured_capacity_mAh_g),
        "measured_c_rate": float(measured_c_rate),
        "voltage_window": voltage_window,
        "notes": notes,
        "source_recipe_path": recipe_path,
    }

def apply_overrides(best_detail: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply feedback-informed overrides (if present) to the literature-extracted best_detail.
    """
    out = dict(best_detail)
    # optional override fields exist only in decision_next mode
    frac = decision.get("next_doping_fraction", None)
    if isinstance(frac, (int, float)):
        out["doping_fraction"] = float(frac)
    txt = decision.get("next_doping_level_text", None)
    if isinstance(txt, str) and txt.strip():
        out["doping_level_text"] = txt.strip()
    basis = decision.get("next_doping_basis", None)
    if isinstance(basis, str) and basis.strip():
        out["doping_basis"] = basis.strip()
    mmode = decision.get("next_modifier_mode", None)
    if isinstance(mmode, str) and mmode.strip():
        out["modifier_mode"] = mmode.strip()
    precs = decision.get("next_dopant_precursors", None)
    if isinstance(precs, list) and [str(p).strip() for p in precs if str(p).strip()]:
        out["dopant_precursors"] = [str(p).strip() for p in precs if str(p).strip()]
        inf = out.get("inferred_fields") or []
        inf.append("dopant_precursors_overridden_by_feedback_loop")
        out["inferred_fields"] = list(dict.fromkeys(inf))
    adj = decision.get("protocol_adjustments", None)
    if isinstance(adj, list):
        out["protocol_adjustments"] = [str(a).strip() for a in adj if str(a).strip()]
    refl = decision.get("reflection_summary", None)
    if isinstance(refl, str) and refl.strip():
        out["feedback_reflection_summary"] = refl.strip()
    return out
