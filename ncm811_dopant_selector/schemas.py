from __future__ import annotations

from typing import Any, Dict

def schema_scan_candidates() -> Dict[str, Any]:
    """
    Output of SCAN: candidate dopant signatures and evidence snippets with chunk ids.
    """
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dopant_elements": {"type": "array", "items": {"type": "string"}},
                        "dopant_signature": {"type": "string"},
                        "modifier_mode": {"type": "string"},  # bulk_TM_substitution | Li_site_substitution | surface_coating | unknown
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                            },
                        },
                        "notes": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "global_notes": {"type": "string"},
        },
    }

def schema_candidate_detail() -> Dict[str, Any]:
    """
    Output of DETAIL: a single candidate with doping levels, precursor formulas, performance evidence,
    and structured synthesis/test parameters useful for transferability reasoning.

    NOTE: We deliberately keep this schema "broad-but-light":
    - numeric fields are nullable (null if not explicitly supported)
    - categorical fields may be inferred, but must be marked in inferred_fields
    """
    return {
        "type": "object",
        "properties": {
            "dopant_signature": {"type": "string"},
            "dopant_elements": {"type": "array", "items": {"type": "string"}},
            "modifier_mode": {"type": "string"},

            # Doping level
            "doping_level_text": {"type": ["string", "null"]},
            "doping_fraction": {"type": ["number", "null"]},
            "doping_basis": {"type": "string"},  # TM | Li | wt_fraction | unknown

            # Precursors (may be inferred for hydro/solvothermal compatibility)
            "dopant_precursors": {"type": "array", "items": {"type": "string"}},

            # Performance evidence (INITIAL/FIRST discharge capacity focus)
            "initial_discharge_capacity": {
                "type": "object",
                "properties": {
                    "trend": {"type": "string"},  # improved|degraded|unclear|reported
                    "doped_value_mAh_g": {"type": ["number", "null"]},
                    "baseline_value_mAh_g": {"type": ["number", "null"]},
                    "delta_mAh_g": {"type": ["number", "null"]},
                    "c_rate": {"type": ["number", "null"]},
                    "voltage_window": {"type": ["string", "null"]},
                    "temperature_C": {"type": ["number", "null"]},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                    "quotes": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },

            # Synthesis extraction (verbatim + structured)
            "reported_synthesis": {
                "type": "object",
                "properties": {
                    "method_tags": {"type": "array", "items": {"type": "string"}},
                    "parameters": {"type": "array", "items": {"type": "string"}},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                    "quotes": {"type": "array", "items": {"type": "string"}},
                },
            },

            # Structured synthesis/test parameters for transfer reasoning (null if unknown)
            "reported_synthesis_struct": {
                "type": "object",
                "properties": {
                    "primary_route": {"type": ["string", "null"]},  # hydrothermal/solvothermal/co-precipitation/sol-gel/solid-state/surface_coating/unknown
                    "dopant_introduction_stage": {"type": ["string", "null"]},  # e.g., during_precursor, during_hydrothermal, post_coating, during_calcination, unknown
                    "solvent_system": {"type": ["string", "null"]},
                    "chelating_agent": {"type": ["string", "null"]},
                    "base_or_pH_agent": {"type": ["string", "null"]},
                    "pH": {"type": ["number", "null"]},
                    "hydro_T_C": {"type": ["number", "null"]},
                    "hydro_time_h": {"type": ["number", "null"]},
                    "aging_time_h": {"type": ["number", "null"]},
                    "drying_T_C": {"type": ["number", "null"]},
                    "calc1_T_C": {"type": ["number", "null"]},
                    "calc1_time_h": {"type": ["number", "null"]},
                    "calc2_T_C": {"type": ["number", "null"]},
                    "calc2_time_h": {"type": ["number", "null"]},
                    "calc_atmosphere": {"type": ["string", "null"]},
                    "li_excess_fraction": {"type": ["number", "null"]},
                    "particle_morphology_notes": {"type": ["string", "null"]},
                },
            },

            # Mechanistic cues from the paper(s) (helps interaction reasoning)
            "mechanistic_evidence": {
                "type": "object",
                "properties": {
                    "claims": {"type": "array", "items": {"type": "string"}},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                    "quotes": {"type": "array", "items": {"type": "string"}},
                },
            },

            # Inference flags
            "inferred_fields": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "overall_confidence": {"type": "number"},
        },
    }

def schema_decision() -> Dict[str, Any]:
    """
    Output of DECIDE: picks the best candidate and justifies it with interaction-aware reasoning.
    We keep 'score' for plotting compatibility, but add sub-scores and expected outcomes.
    """
    return {
        "type": "object",
        "properties": {
            "best_dopant_signature": {"type": "string"},
            "reasoning_summary": {"type": "string"},
            "global_interaction_factors_considered": {"type": "array", "items": {"type": "string"}},
            "recommended_protocol_adjustments": {"type": "array", "items": {"type": "string"}},
            "ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dopant_signature": {"type": "string"},
                        "score": {"type": "number"},
                        "evidence_strength_score": {"type": "number"},
                        "transferability_score": {"type": "number"},
                        "mechanistic_plausibility_score": {"type": "number"},
                        "practical_feasibility_score": {"type": "number"},
                        "expected_delta_mAh_g": {"type": ["number", "null"]},
                        "expected_initial_mAh_g": {"type": ["number", "null"]},
                        "key_interactions": {"type": "array", "items": {"type": "string"}},
                        "mismatch_risks": {"type": "array", "items": {"type": "string"}},
                        "why": {"type": "string"},
                    },
                },
            },
            "confidence": {"type": "number"},
            "known_gaps": {"type": "array", "items": {"type": "string"}},
        },
    }

def schema_protocol() -> Dict[str, Any]:
    """
    Output of PROTOCOL: a lab-ready hydro/solvothermal protocol, plus rationale & interaction notes.
    """
    return {
        "type": "object",
        "properties": {
            "protocol_title": {"type": "string"},
            "overview": {"type": "string"},
            "scientific_rationale": {"type": "string"},
            "transferability_notes": {"type": "array", "items": {"type": "string"}},
            "expected_outcome": {
                "type": "object",
                "properties": {
                    "expected_initial_discharge_mAh_g": {"type": ["number", "null"]},
                    "expected_delta_mAh_g": {"type": ["number", "null"]},
                    "uncertainty_notes": {"type": "string"},
                },
            },
            "weighing_table_notes": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "critical_controls": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "safety_notes": {"type": "array", "items": {"type": "string"}},
            "quality_checks": {"type": "array", "items": {"type": "string"}},
        },
    }

def schema_reflection_plan() -> Dict[str, Any]:
    """
    Output of REFLECT/PLAN: how to adapt retrieval + selection based on lab feedback.
    """
    return {
        "type": "object",
        "properties": {
            "reflection_summary": {"type": "string"},
            "lessons": {"type": "array", "items": {"type": "string"}},
            "hypotheses": {"type": "array", "items": {"type": "string"}},
            "next_search_queries": {"type": "array", "items": {"type": "string"}},
            "avoid_exact_recipes": {"type": "array", "items": {"type": "string"}},
            "exploration_strategy": {"type": "string"},
        },
    }

def schema_decision_next() -> Dict[str, Any]:
    """
    Output of DECIDE (feedback-aware): chooses next best candidate and (optionally) overrides
    dopant fraction / precursor choices for the next lab iteration.

    Enhanced with interaction-aware fields to help steer iterative optimization.
    """
    return {
        "type": "object",
        "properties": {
            "best_dopant_signature": {"type": "string"},
            "reasoning_summary": {"type": "string"},
            "reflection_summary": {"type": "string"},
            "next_doping_level_text": {"type": ["string", "null"]},
            "next_doping_fraction": {"type": ["number", "null"]},
            "next_doping_basis": {"type": "string"},
            "next_modifier_mode": {"type": "string"},
            "next_dopant_precursors": {"type": "array", "items": {"type": "string"}},
            "protocol_adjustments": {"type": "array", "items": {"type": "string"}},
            "parameter_sensitivity_hypotheses": {"type": "array", "items": {"type": "string"}},
            "do_not_repeat_exact_recipes": {"type": "array", "items": {"type": "string"}},
            "ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dopant_signature": {"type": "string"},
                        "score": {"type": "number"},
                        "evidence_strength_score": {"type": "number"},
                        "transferability_score": {"type": "number"},
                        "mechanistic_plausibility_score": {"type": "number"},
                        "practical_feasibility_score": {"type": "number"},
                        "expected_delta_mAh_g": {"type": ["number", "null"]},
                        "expected_initial_mAh_g": {"type": ["number", "null"]},
                        "mismatch_risks": {"type": "array", "items": {"type": "string"}},
                        "why": {"type": "string"},
                    },
                },
            },
            "confidence": {"type": "number"},
            "known_gaps": {"type": "array", "items": {"type": "string"}},
        },
    }

def schema_mechanism_map() -> Dict[str, Any]:
    """
    Output schema for a dopant -> mechanism-category map.
    This is used for a publication-style figure (heatmap / mechanism map).

    NOTE: Categories are fixed to MECH_CATEGORIES to keep plots comparable across runs.
    """
    return {
        "type": "object",
        "properties": {
            "categories": {"type": "array", "items": {"type": "string"}},
            "dopants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dopant_signature": {"type": "string"},
                        "overall_expected_direction": {"type": "string"},  # positive|negative|unclear
                        "category_links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "direction": {"type": "string"},  # positive|negative|unclear
                                    "strength": {"type": "number"},   # 0..1
                                    "evidence": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        "notes": {"type": "string"},
                    },
                },
            },
        },
    }
