from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from .constants import HOST_SET, _ATOMIC_WEIGHT, _ELEMENT_SYMBOLS
from .models import SynthesisRow
from .utils import clean_element_symbol, parse_dopant_signature


class FormulaError(ValueError):
    pass

def _tokenize_formula(s: str) -> List[str]:
    s = s.replace(" ", "")
    # Standardize hydrate separators
    s = s.replace("·", "+").replace("∙", "+").replace(".", "+")
    # Remove charge annotations
    s = re.sub(r"[\+\-]\d*$", "", s)
    # Keep parentheses and + as separators
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "()+":
            tokens.append(ch)
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif ch.isalpha():
            # element symbol
            if i + 1 < len(s) and s[i+1].islower():
                tokens.append(s[i:i+2])
                i += 2
            else:
                tokens.append(s[i])
                i += 1
        else:
            i += 1
    return tokens

def parse_formula_counts(formula: str) -> Dict[str, float]:
    """
    Parse a chemical formula into element counts.
    Supports hydrates via '·' or '.' or '+'.
    Supports parentheses.
    Supports leading coefficients like '6H2O'.
    Returns counts as floats (for safety with decimal stoich; rarely used here).
    """
    if not formula or not str(formula).strip():
        raise FormulaError("Empty formula.")
    tokens = _tokenize_formula(str(formula))
    # Split by '+'
    parts: List[List[str]] = []
    cur: List[str] = []
    for tok in tokens:
        if tok == "+":
            if cur:
                parts.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        parts.append(cur)

    total: Dict[str, float] = {}
    for part in parts:
        counts = _parse_tokens_counts(part)
        for el, c in counts.items():
            total[el] = total.get(el, 0.0) + c
    return total

def _parse_tokens_counts(tokens: List[str]) -> Dict[str, float]:
    # Handle leading coefficient
    mult = 1.0
    if tokens and re.fullmatch(r"\d+(?:\.\d+)?", tokens[0]):
        mult = float(tokens[0])
        tokens = tokens[1:]

    stack: List[Dict[str, float]] = [dict()]
    i = 0

    def add(el: str, n: float):
        stack[-1][el] = stack[-1].get(el, 0.0) + n

    def read_num(j: int) -> Tuple[float, int]:
        if j < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[j]):
            return float(tokens[j]), j + 1
        return 1.0, j

    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            stack.append(dict())
            i += 1
        elif tok == ")":
            i += 1
            n, i = read_num(i)
            grp = stack.pop()
            for el, c in grp.items():
                add(el, c * n)
        else:
            # element
            if tok not in _ELEMENT_SYMBOLS:
                # if token is garbage, try to clean element symbol
                cleaned = clean_element_symbol(tok)
                if not cleaned:
                    raise FormulaError(f"Unknown element token: {tok}")
                tok = cleaned
            i += 1
            n, i = read_num(i)
            add(tok, n)

    if len(stack) != 1:
        raise FormulaError("Unbalanced parentheses in formula.")
    out = {el: c * mult for el, c in stack[0].items()}
    return out

def molar_mass(formula: str) -> float:
    counts = parse_formula_counts(formula)
    mm = 0.0
    for el, n in counts.items():
        aw = _ATOMIC_WEIGHT.get(el)
        if aw is None:
            raise FormulaError(f"Missing atomic weight for element: {el}")
        mm += aw * n
    return mm

def atoms_per_formula(formula: str, element: str) -> float:
    counts = parse_formula_counts(formula)
    return float(counts.get(element, 0.0))

def ncm811_base_stoich() -> Dict[str, float]:
    # LiNi0.8Co0.1Mn0.1O2
    return {"Li": 1.0, "Ni": 0.8, "Co": 0.1, "Mn": 0.1, "O": 2.0}

def compute_weighing_table(
    target_mass_g: float,
    li_excess_fraction: float,
    target: SynthesisRow,
    best_detail: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic grams computation.

    Assumptions:
    - target_mass_g refers to final oxide mass (approx).
    - dopant_mode determines basis:
      * bulk_TM_substitution: dopant replaces TM fraction; Ni/Co/Mn scaled down proportionally.
      * Li_site_substitution: dopant replaces Li; Li reduced accordingly.
      * surface_coating / unknown: treat as wt_fraction on top of base NCM811 mass.
    """
    if target_mass_g <= 0:
        raise ValueError("target_batch_mass_g must be > 0")

    base = ncm811_base_stoich()
    dop_sig = str(best_detail.get("dopant_signature", "")).strip()
    dop_elems = [clean_element_symbol(x) for x in (best_detail.get("dopant_elements") or [])]
    dop_elems = [x for x in dop_elems if x and x not in HOST_SET]
    if not dop_elems and dop_sig:
        dop_elems = parse_dopant_signature(dop_sig)
    mode = str(best_detail.get("modifier_mode", "unknown"))
    basis = str(best_detail.get("doping_basis", "unknown"))
    frac = best_detail.get("doping_fraction", None)
    frac = float(frac) if isinstance(frac, (int, float)) else None

    # Infer fraction if missing: choose conservative default and mark
    inferred = []
    if frac is None:
        # default: 0.005 (0.5 mol%) total dopant fraction
        frac = 0.005
        inferred.append("doping_fraction_defaulted_to_0.005")

    if frac < 0:
        frac = abs(frac)
    if frac > 0.2:
        # extremely high; cap
        frac = 0.2
        inferred.append("doping_fraction_capped_at_0.2")

    # Determine dopant split across elements
    # If co-doping and no individual split is known, split evenly.
    if dop_elems:
        per = frac / len(dop_elems)
        dop_fracs = {el: per for el in dop_elems}
    else:
        dop_fracs = {}

    # Build doped stoichiometry
    sto = dict(base)  # includes O
    if mode == "bulk_TM_substitution" or (mode == "unknown" and basis in {"TM", "unknown"}):
        x = sum(dop_fracs.values())
        # scale down Ni/Co/Mn proportionally
        scale = max(0.0, 1.0 - x)
        sto["Ni"] = base["Ni"] * scale
        sto["Co"] = base["Co"] * scale
        sto["Mn"] = base["Mn"] * scale
        for el, f in dop_fracs.items():
            sto[el] = f
        # Li stays 1
    elif mode == "Li_site_substitution" or basis == "Li":
        x = sum(dop_fracs.values())
        sto["Li"] = max(0.0, base["Li"] * (1.0 - x))
        for el, f in dop_fracs.items():
            sto[el] = f
        # TM unchanged
    else:
        # surface coating: compute base NCM811 grams and dopant grams separately
        # Interpret frac as wt_fraction of coating relative to base mass.
        sto = dict(base)
        inferred.append("surface_coating_mass_fraction_assumed")

    # Apply Li excess
    sto_li = sto.get("Li", base["Li"])
    sto["Li"] = sto_li * (1.0 + li_excess_fraction)

    # Compute formula weight (approx)
    fw = 0.0
    for el, n in sto.items():
        aw = _ATOMIC_WEIGHT.get(el)
        if aw is None:
            raise FormulaError(f"Missing atomic weight for element {el} for formula weight.")
        fw += aw * n

    moles_fu = target_mass_g / fw

    # Moles of each element to supply
    moles_el = {el: n * moles_fu for el, n in sto.items() if el != "O"}  # oxygen supplied by calcination/air

    # Precursor formulas from synthesis.csv
    prec_map = {
        "Li": target.li_precursor,
        "Ni": target.ni_precursor,
        "Co": target.co_precursor,
        "Mn": target.mn_precursor,
    }

    # Dopant precursors (from GPT), try map by element if possible
    dop_prec_list = best_detail.get("dopant_precursors") or []
    dop_prec_list = [str(x).strip() for x in dop_prec_list if str(x).strip()]

    # A light heuristic mapping for dopant precursors:
    # If only one dopant and multiple precursors, pick the first that contains the element token.
    dop_prec_map: Dict[str, str] = {}
    for el in dop_elems:
        chosen = None
        for p in dop_prec_list:
            if clean_element_symbol(p) == el or re.search(rf"\b{el}\b", p):
                chosen = p
                break
        if chosen is None and dop_prec_list:
            chosen = dop_prec_list[0]
        if chosen:
            dop_prec_map[el] = chosen

    # Compute masses for host precursors
    weighing_rows: List[Dict[str, Any]] = []
    for el in ["Li","Ni","Co","Mn"]:
        prec = prec_map.get(el, "")
        if not prec:
            continue
        try:
            mm = molar_mass(prec)
            a = atoms_per_formula(prec, el)
            if a <= 0:
                raise FormulaError(f"Precursor {prec} does not contain {el}.")
            mol_prec = moles_el[el] / a
            g = mol_prec * mm
            weighing_rows.append({
                "element": el,
                "precursor": prec,
                "moles_element": moles_el[el],
                "moles_precursor": mol_prec,
                "molar_mass_g_mol": mm,
                "grams": g,
                "source": "computed_from_synthesis_csv",
            })
        except Exception as e:
            weighing_rows.append({
                "element": el,
                "precursor": prec,
                "moles_element": moles_el.get(el),
                "grams": None,
                "source": "needs_manual_or_gpt_normalization",
                "error": str(e),
            })

    # Compute masses for dopant precursors
    for el in dop_elems:
        mol_el = moles_el.get(el)
        if mol_el is None:
            continue
        prec = dop_prec_map.get(el) or (dop_prec_list[0] if dop_prec_list else "")
        if not prec:
            weighing_rows.append({
                "element": el,
                "precursor": None,
                "moles_element": mol_el,
                "grams": None,
                "source": "missing_dopant_precursor",
            })
            continue
        try:
            mm = molar_mass(prec)
            a = atoms_per_formula(prec, el)
            if a <= 0:
                raise FormulaError(f"Dopant precursor {prec} does not contain {el}.")
            mol_prec = mol_el / a
            g = mol_prec * mm
            weighing_rows.append({
                "element": el,
                "precursor": prec,
                "moles_element": mol_el,
                "moles_precursor": mol_prec,
                "molar_mass_g_mol": mm,
                "grams": g,
                "source": "computed_from_dopant_fraction",
            })
        except Exception as e:
            weighing_rows.append({
                "element": el,
                "precursor": prec,
                "moles_element": mol_el,
                "grams": None,
                "source": "needs_manual_or_gpt_normalization",
                "error": str(e),
            })

    # Summaries
    out = {
        "target_batch_mass_g": target_mass_g,
        "li_excess_fraction": li_excess_fraction,
        "doped_stoichiometry": sto,
        "formula_weight_g_mol": fw,
        "moles_formula_units": moles_fu,
        "weighing_rows": weighing_rows,
        "inferred": inferred,
    }
    return out
