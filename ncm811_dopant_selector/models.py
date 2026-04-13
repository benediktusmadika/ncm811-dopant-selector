from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    path: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

@dataclass
class ModelCapabilityProfile:
    raw_model: str
    normalized_model: str
    supports_reasoning_config: bool
    supports_reasoning_none: bool
    only_high_reasoning: bool
    temperature_supported: bool
    temperature_requires_reasoning_none: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_model": self.raw_model,
            "normalized_model": self.normalized_model,
            "supports_reasoning_config": self.supports_reasoning_config,
            "supports_reasoning_none": self.supports_reasoning_none,
            "only_high_reasoning": self.only_high_reasoning,
            "temperature_supported": self.temperature_supported,
            "temperature_requires_reasoning_none": self.temperature_requires_reasoning_none,
        }

@dataclass
class GenerationRequestSettings:
    temperature: Optional[float]
    reasoning_effort: Optional[str]

@dataclass
class ResponseEnvelope:
    raw_text: str
    status: Optional[str]
    incomplete_reason: Optional[str]
    refusal_text: Optional[str]

@dataclass
class SynthesisRow:
    row_id: int
    method: str  # hydrothermal/solvothermal/unknown
    li_precursor: str
    ni_precursor: str
    co_precursor: str
    mn_precursor: str
    hydro_T_C: Optional[float]
    hydro_time_h: Optional[float]
    calc1_T_C: Optional[float]
    calc1_time_h: Optional[float]
    calc2_T_C: Optional[float]
    calc2_time_h: Optional[float]
    c_rate: Optional[float]
    baseline_capacity_mAh_g: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabFeedback:
    """
    A single lab trial result used for closed-loop improvement.

    measured_initial_discharge_mAh_g: the FIRST/INITIAL discharge capacity (mAh/g),
    ideally at 0.1C (default in this workflow).
    """
    trial_id: int
    timestamp: str
    synthesis_row_id: int
    dopant_signature: str
    doping_fraction: Optional[float]
    doping_basis: str
    modifier_mode: str
    dopant_precursors: List[str]
    measured_initial_discharge_mAh_g: float
    measured_c_rate: float = 0.1
    voltage_window: Optional[str] = None
    notes: Optional[str] = None
    source_recipe_path: Optional[str] = None

    def recipe_key(self) -> str:
        # Key for "exact recipe" identity checks
        f = "null" if self.doping_fraction is None else f"{float(self.doping_fraction):.6f}"
        return f"{self.dopant_signature}|{self.doping_basis}|{f}|{self.modifier_mode}"
