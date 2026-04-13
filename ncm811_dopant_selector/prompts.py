from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple

from .constants import HOST_ELEMENTS, MECH_CATEGORIES
from .models import Chunk, SynthesisRow


def prompt_mechanism_map(
    target: "SynthesisRow",
    ranking: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]],
    top_k: int = 8,
) -> Tuple[str, str]:
    """
    Ask GPT to convert mechanistic evidence into a compact set of mechanism categories.
    This is strictly for interpretability/visualization; it does NOT change the best-recipe selection.
    """
    top = (ranking or [])[:max(1, top_k)]
    by_sig = {str(c.get("dopant_signature") or ""): c for c in (candidate_details or [])}

    items = []
    for r in top:
        sig = str(r.get("dopant_signature") or "").strip()
        det = by_sig.get(sig, {})
        mech = det.get("mechanistic_evidence") or {}
        synth = det.get("reported_synthesis_struct") or {}
        cap = det.get("initial_discharge_capacity") or {}
        items.append({
            "dopant_signature": sig,
            "subscores": {
                "evidence_strength_score": r.get("evidence_strength_score"),
                "transferability_score": r.get("transferability_score"),
                "mechanistic_plausibility_score": r.get("mechanistic_plausibility_score"),
                "practical_feasibility_score": r.get("practical_feasibility_score"),
            },
            "expected": {
                "expected_delta_mAh_g": r.get("expected_delta_mAh_g"),
                "expected_initial_mAh_g": r.get("expected_initial_mAh_g"),
            },
            "key_interactions": r.get("key_interactions") or [],
            "mismatch_risks": r.get("mismatch_risks") or [],
            "mechanistic_claims": mech.get("claims") or [],
            "mechanistic_quotes": (mech.get("quotes") or [])[:4],
            "synthesis_struct": synth,
            "capacity_context": {
                "trend": cap.get("trend"),
                "delta_mAh_g": cap.get("delta_mAh_g"),
                "c_rate": cap.get("c_rate"),
                "voltage_window": cap.get("voltage_window"),
            },
        })

    sys = (
        "You are a battery materials scientist creating an interpretable mechanism map for a Nature-style figure.\n"
        "Host constraint reminder: host elements are ONLY Li, Ni, Co, Mn, O. All other elements are dopants.\n"
        "Your job is to map each dopant candidate to a SMALL set of fixed mechanism categories.\n"
        "Do not invent new categories; use ONLY the provided category list.\n"
        "If evidence is weak/ambiguous, output direction='unclear' and strength near 0.\n"
        "Think carefully about how the mechanism could influence INITIAL/FIRST discharge capacity.\n"
        "Output ONLY JSON.\n"
    )

    user = (
        "Goal: Build a mechanism map for dopant candidates ranked for improving INITIAL/FIRST discharge capacity.\n\n"
        f"Fixed categories (use only these): {MECH_CATEGORIES}\n\n"
        "For each dopant, output:\n"
        "- overall_expected_direction: positive|negative|unclear for initial capacity\n"
        "- category_links: subset of categories with direction and strength (0..1)\n"
        "- Provide short evidence strings (quotes or paraphrases from inputs).\n\n"
        "Target baseline synthesis context (for transfer reasoning):\n"
        f"- method={target.method}, hydro_T_C={target.hydro_T_C}, hydro_time_h={target.hydro_time_h}, "
        f"calc1_T_C={target.calc1_T_C}, calc2_T_C={target.calc2_T_C}, c_rate={target.c_rate}\n\n"
        "Candidates (top-ranked):\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON matching the schema."
    )
    return sys, user

def prompt_scan(chunks: List[Chunk], target_method: str) -> Tuple[str, str]:
    """
    SCAN prompt: identify dopant candidates and evidence chunks.
    """
    sys = (
        "You are an expert materials scientist and careful evidence extractor.\n"
        "You MUST follow the host constraint: host elements are ONLY Li, Ni, Co, Mn, O. Any other element counts as a dopant/modifier.\n"
        "You may see charged ions like La3+, Sr2+; normalize them to element symbols.\n"
        "Goal: find candidate dopant elements (including co-doping) used with NCM811 and identify supporting quotes.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )

    # Provide chunk list with ids
    ctx = "\n\n".join([f"[CHUNK {c.chunk_id}]\n{c.text}" for c in chunks])

    user = (
        f"Task: SCAN the corpus chunks for dopant/modifier candidates relevant to NCM811 with {target_method} synthesis.\n"
        "Return a list of candidates. Each candidate must have:\n"
        "- dopant_elements: list of element symbols (NOT host elements)\n"
        "- dopant_signature: sorted '+'-joined signature (e.g., 'Al+La')\n"
        "- modifier_mode: one of 'bulk_TM_substitution','Li_site_substitution','surface_coating','unknown'\n"
        "- evidence: 1-5 items with chunk_id and an exact short quote supporting dopant existence\n"
        "- confidence: 0..1\n"
        "Important: include candidates even if precursor/amount is not present in these chunks.\n"
        "Context chunks:\n"
        f"{ctx}\n"
    )
    return sys, user

def prompt_detail(candidate_sig: str, target: SynthesisRow, chunks: List[Chunk]) -> Tuple[str, str]:
    """
    DETAIL prompt: extract candidate record with synthesis/test parameters that matter for transferability.

    Design goals:
    - Evidence-grounded extraction for numbers and specific conditions
    - Allow careful inference for operationally necessary fields (e.g., soluble precursor choice),
      but explicitly mark those in inferred_fields
    - Capture enough synthesis "state variables" to reason about interactions later
    """
    sys = (
        "You are an expert materials scientist extracting structured experimental details from paper text.\n"
        "Host constraint: host elements are ONLY Li, Ni, Co, Mn, O. Any other element is a dopant/modifier.\n"
        "Normalize ions like 'La3+' -> 'La'.\n"
        "You are extracting evidence for a CLOSED-LOOP optimization campaign. The decision step will condition on a target synthesis baseline.\n"
        "\nEvidence rules:\n"
        "- For EVERY numeric value (capacity, temperature, time, pH, fraction, etc.), include evidence_chunks ids AND quotes.\n"
        "- If a numeric value is not explicitly supported, set it to null (do NOT guess).\n"
        "- For categorical operational choices that are required to execute a hydro/solvothermal route (e.g., dopant precursor salt),\n"
        "  you MAY propose a best-compatible option but you MUST:\n"
        "  (a) include it in inferred_fields, and (b) justify compatibility briefly in warnings.\n"
        "\nOutput ONLY JSON.\n"
    )
    ctx = "\n\n".join([f"[CHUNK {c.chunk_id}]\n{c.text}" for c in chunks])

    user = (
        f"Task: Build a detailed candidate record for dopant_signature='{candidate_sig}'.\n\n"
        f"TARGET baseline row (from synthesis.csv):\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        "You must extract or infer the following fields (conforming to the schema):\n"
        "1) Doping specification\n"
        "   - doping_level_text (exact text if present)\n"
        "   - doping_fraction (decimal fraction; total dopant on its basis)\n"
        "   - doping_basis (TM|Li|wt_fraction|unknown)\n"
        "   - dopant_precursors (list). If missing, propose soluble precursors compatible with target hydro/solvothermal chemistry and mark inferred_fields.\n"
        "\n"
        "2) Performance evidence (objective)\n"
        "   - initial_discharge_capacity: extract doped/baseline values if present, with c_rate (0.1C synonyms) and voltage window.\n"
        "   - If numeric values are not present, extract the strongest qualitative statement about initial/first discharge capacity.\n"
        "\n"
        "3) Reported synthesis + structured parameters for transfer reasoning\n"
        "   - reported_synthesis.method_tags and reported_synthesis.parameters (as strings)\n"
        "   - reported_synthesis_struct: fill any known values (primary_route, hydro_T_C, calcination temps/times, pH, solvent, atmosphere, Li excess, etc.)\n"
        "   - If unknown, set null; do not guess numbers.\n"
        "\n"
        "4) Mechanistic evidence\n"
        "   - mechanistic_evidence.claims: extract mechanistic claims linked to capacity (e.g., cation mixing suppression, oxygen loss mitigation, surface reconstruction, microcracks)\n"
        "   - Provide quotes and chunk ids.\n"
        "5) overall_confidence\n"
        "   - Provide a single 0..1 overall confidence score for the entire candidate record.\n"
        "\n"
        "Context chunks:\n"
        f"{ctx}\n"
    )
    return sys, user

def prompt_decide(target: SynthesisRow, candidate_details: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    DECIDE prompt: choose best dopant conditioned on target synthesis row.
    Enhanced: explicit interaction-aware transferability reasoning, not just "highest literature number".
    """
    sys = (
        "You are a senior battery materials PI selecting a single best dopant strategy for NCM811.\n"
        "Primary objective: maximize INITIAL/FIRST discharge capacity (mAh/g) at ~0.1C.\n"
        "Host constraint: host elements only Li, Ni, Co, Mn, O. Any other element is dopant/modifier.\n"
        "\nYou MUST reason about transferability:\n"
        "- Literature improvements are condition-dependent.\n"
        "- If a dopant shows high initial capacity under a different synthesis route/parameters than the target baseline, you must discount or adapt it.\n"
        "\nYou must consider interactions between:\n"
        "- dopant identity / site / mode (bulk TM substitution vs Li-site vs coating)\n"
        "- dopant level and precursor chemistry/solubility\n"
        "- hydro/solvothermal parameters (T, time, pH, solvent, additives/chelants)\n"
        "- calcination profile (T/time/atmosphere, Li excess) affecting dopant incorporation, cation mixing, surface reconstruction\n"
        "- test conditions (C-rate, voltage window) and how they affect reported initial capacity\n"
        "\nUse a structured scoring rubric and be explicit about uncertainty.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )
    user = (
        f"TARGET baseline synthesis row (from synthesis.csv):\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        f"CANDIDATE DETAILS (from corpus-level RAG+extraction across ALL PDFs):\n{json.dumps(candidate_details, ensure_ascii=False)}\n\n"
        "Task:\n"
        "1) For each candidate, compute sub-scores (0..100):\n"
        "   - evidence_strength_score: strength of evidence that initial/first discharge capacity improves (numeric paired baseline>doped is strongest)\n"
        "   - transferability_score: expected transfer to the TARGET baseline given synthesis/test condition match and plausible adaptation\n"
        "   - mechanistic_plausibility_score: does the mechanism plausibly increase initial capacity under target conditions?\n"
        "   - practical_feasibility_score: can we execute using hydro/solvothermal workflow + available precursor classes?\n"
        "2) Combine to a total 'score' (0..100) and rank.\n"
        "3) Pick ONE best_dopant_signature.\n"
        "4) Provide expected_delta_mAh_g and expected_initial_mAh_g estimates when possible.\n"
        "   - If numeric estimates are too uncertain, set them null and explain in mismatch_risks/known_gaps.\n"
        "5) Provide global_interaction_factors_considered (bullet list), and recommended_protocol_adjustments (how to adapt baseline to realize the dopant benefit).\n"
        "\nImportant:\n"
        "- Do NOT just pick the candidate with the highest literature capacity number if synthesis/test conditions differ.\n"
        "- If candidate evidence is only qualitative, it can still win if transferability and feasibility are high.\n"
        "- If a candidate is a surface coating (not bulk doping), be explicit about why it affects initial discharge capacity (often it affects cycling more than initial).\n"
        "- Always treat 'initial' and 'first' discharge capacity as the same objective.\n"
    )
    return sys, user

def prompt_protocol(target: SynthesisRow, best_detail: Dict[str, Any], weighing_table: Dict[str, Any]) -> Tuple[str, str]:
    """
    PROTOCOL prompt: write lab-ready hydro/solvothermal synthesis protocol using baseline conditions,
    explicitly addressing parameter interactions and transferability.
    """
    sys = (
        "You are an expert experimentalist writing a lab-ready synthesis protocol for doped NCM811.\n"
        "The goal is to MAXIMIZE INITIAL/FIRST discharge capacity (mAh/g) at ~0.1C.\n"
        "\nKey requirement:\n"
        "- You MUST use the baseline synthesis row as the default process.\n"
        "- If the literature evidence suggests the dopant benefit depends on parameters that differ from the baseline,\n"
        "  you must propose minimal, justified adjustments and clearly list them in assumptions and transferability_notes.\n"
        "\nWrite for a graduate student to execute.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )
    user = (
        "Write a detailed hydrothermal/solvothermal synthesis procedure that matches the baseline row conditions, "
        "and incorporates the chosen dopant strategy.\n\n"
        f"Baseline synthesis row (synthesis.csv):\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        f"Chosen dopant detail (from corpus evidence + inference):\n{json.dumps(best_detail, ensure_ascii=False)}\n\n"
        f"Computed weighing table (grams):\n{json.dumps(weighing_table, ensure_ascii=False)}\n\n"
        "Output requirements:\n"
        "- Include a clear scientific_rationale that links dopant mechanism to INITIAL discharge capacity and explains why it should transfer to baseline conditions.\n"
        "- Include transferability_notes describing which parameters are most sensitive and how to keep them controlled.\n"
        "- Include expected_outcome as a cautious estimate (use null if too uncertain) and explain uncertainty.\n"
        "- Provide fully specified steps with temperatures, times, solvents, pH strategy (if applicable), washing/drying, and calcination profile.\n"
        "- Provide critical_controls that directly protect initial capacity (e.g., Li/Ni mixing control, oxygen loss control, dopant homogeneity, moisture/CO2 control).\n"
        "- Provide quality_checks (XRD, SEM/EDS, ICP, BET/tap density as appropriate) to validate dopant incorporation.\n"
        "- If some details are not in evidence, choose reasonable defaults and list them in assumptions.\n"
    )
    return sys, user

def prompt_reflection_plan(target: SynthesisRow, feedback_ctx: Dict[str, Any]) -> Tuple[str, str]:
    """
    REFLECT/PLAN prompt: use lab feedback to propose new retrieval queries and selection guidance.
    Enhanced: explicitly connect outcomes to synthesis-parameter interactions, not only dopant identity.
    """
    sys = (
        "You are a senior battery materials PI running a closed-loop optimization campaign.\n"
        "Objective: maximize INITIAL/FIRST discharge capacity (mAh/g) at ~0.1C.\n"
        "Host constraint: host elements are ONLY Li, Ni, Co, Mn, O; any other element is a dopant/modifier.\n"
        "You will be shown a baseline synthesis row and a log of previous lab trials with measured outcomes.\n"
        "\nCritical requirement:\n"
        "- Learn not only from the measured capacity value, but from the interaction of (dopant, doping level, precursor chemistry, synthesis parameters, calcination, and test conditions).\n"
        "- If a trial underperformed, hypothesize whether it was due to synthesis mismatch (incorporation, segregation, particle morphology, oxygen loss, cation mixing) rather than the dopant being 'bad'.\n"
        "- Suggest actionable next experiments and retrieval queries that target these interactions.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )
    user = (
        f"Baseline synthesis row (synthesis.csv):\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        f"Lab feedback context (measured outcomes):\n{json.dumps(feedback_ctx, ensure_ascii=False)}\n\n"
        "Return:\n"
        "- reflection_summary: 4-8 sentences\n"
        "- lessons: 3-10 bullets\n"
        "- hypotheses: 4-12 bullets (mechanistic + synthesis-parameter interactions)\n"
        "- next_search_queries: 6-16 RAG queries to retrieve relevant evidence from the PDF corpus\n"
        "- avoid_exact_recipes: list of exact recipe keys to avoid repeating unless replication is explicitly needed\n"
        "- exploration_strategy: one of 'exploit','explore','hybrid'\n"
        "\nImportant:\n"
        "- Prefer queries that connect initial/first discharge capacity to synthesis parameters (hydro/solvothermal T/time/pH/solvent/additives, calcination profile, Li excess, atmosphere).\n"
        "- Include synonyms: 'first discharge', 'initial capacity', '0.1C', '0.1 C'.\n"
    )
    return sys, user

def prompt_decide_next(
    target: SynthesisRow,
    candidate_details: List[Dict[str, Any]],
    feedback_ctx: Dict[str, Any],
    reflection_plan: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Feedback-aware DECIDE prompt: choose the next experiment.
    Enhanced: explicitly reason about synthesis-parameter interactions and transferability, not only measured capacity.
    """
    sys = (
        "You are a senior battery materials PI selecting the NEXT experiment in a closed-loop campaign.\n"
        "Primary objective: maximize INITIAL/FIRST discharge capacity (mAh/g) at ~0.1C.\n"
        "Host constraint: host elements only Li, Ni, Co, Mn, O.\n"
        "\nYou MUST incorporate feedback with strong reasoning:\n"
        "- Do not overfit to a single number; interpret outcomes in the context of synthesis parameters and how they affect dopant incorporation, defects, and microstructure.\n"
        "- If a previous trial underperformed, consider whether the dopant effect is condition-dependent and propose parameter adjustments or alternative dopant modes.\n"
        "\nUse decomposition + verification:\n"
        "1) Summarize what feedback implies about dopant/parameter interactions.\n"
        "2) Evaluate each candidate with sub-scores (evidence, transferability, mechanism, feasibility).\n"
        "3) Propose the best next experiment: (dopant, level, precursor) + protocol_adjustments.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )
    user = (
        f"TARGET baseline synthesis row:\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        f"LAB FEEDBACK (history):\n{json.dumps(feedback_ctx, ensure_ascii=False)}\n\n"
        f"REFLECTION PLAN (optional):\n{json.dumps(reflection_plan, ensure_ascii=False) if reflection_plan else 'null'}\n\n"
        f"CANDIDATE DETAILS (RAG+extraction over ALL PDFs):\n{json.dumps(candidate_details, ensure_ascii=False)}\n\n"
        "Task:\n"
        "- Choose ONE best_dopant_signature for the NEXT iteration.\n"
        "- Provide next_doping_fraction / next_doping_basis / next_modifier_mode / next_dopant_precursors.\n"
        "- Provide protocol_adjustments (concrete changes to baseline aimed at realizing the dopant benefit).\n"
        "- Provide parameter_sensitivity_hypotheses: which synthesis knobs are likely controlling the initial capacity benefit (and why).\n"
        "- Provide do_not_repeat_exact_recipes based on avoid_exact_recipes from reflection_plan and measured outcomes.\n"
        "- Provide a ranked list with sub-scores and mismatch_risks.\n"
        "\nImportant:\n"
        "- If the best candidate is the same dopant as before, you MUST justify why (e.g., exploring doping fraction, changing introduction stage, tightening controls) rather than blindly repeating.\n"
        "- Prefer candidates where you can explain a plausible mechanistic path to higher initial capacity under the TARGET baseline.\n"
    )
    return sys, user

def prompt_protocol_feedback(
    target: SynthesisRow,
    best_detail: Dict[str, Any],
    weighing_table: Dict[str, Any],
    feedback_ctx: Dict[str, Any],
    decision_next: Dict[str, Any],
) -> Tuple[str, str]:
    """
    PROTOCOL prompt variant that explicitly conditions on feedback + adjustments.
    """
    sys = (
        "You are an expert experimentalist writing a lab-ready synthesis protocol for doped NCM811.\n"
        "You must produce complete operational details.\n"
        "You are in a CLOSED-LOOP campaign; you must incorporate the provided protocol_adjustments derived from feedback.\n"
        "If some details are not in evidence, choose reasonable defaults and list them explicitly in assumptions.\n"
        "Think step-by-step internally, but output ONLY JSON.\n"
    )
    user = (
        "Write a detailed hydrothermal/solvothermal synthesis procedure that matches the baseline row conditions, "
        "and incorporates the chosen dopant strategy.\n\n"
        f"Baseline synthesis row:\n{json.dumps(dataclasses.asdict(target), ensure_ascii=False)}\n\n"
        f"Chosen dopant detail (after overrides):\n{json.dumps(best_detail, ensure_ascii=False)}\n\n"
        f"Weighing table (grams):\n{json.dumps(weighing_table, ensure_ascii=False)}\n\n"
        f"Closed-loop feedback context:\n{json.dumps(feedback_ctx, ensure_ascii=False)}\n\n"
        f"Decision/adjustments:\n{json.dumps(decision_next, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- Use hydrothermal or solvothermal route (match baseline if possible).\n"
        "- Include solvent types, total volumes, concentration targets, pH targets, stirring times, autoclave fill %, heating/cooling profile.\n"
        "- Include drying conditions.\n"
        "- Include calcination steps using baseline temperatures and times; specify atmosphere (assume oxygen or air if not specified) and heating rates.\n"
        "- Include electrode test alignment notes (C-rate=0.1C, voltage window) for fair comparison.\n"
        "- Provide critical_controls and safety_notes.\n"
        "- Anything not supported by evidence must be listed in assumptions.\n"
    )
    return sys, user
