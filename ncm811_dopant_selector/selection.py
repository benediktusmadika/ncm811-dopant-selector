from __future__ import annotations

from typing import Dict, List

from .models import Chunk
from .openai_client import OpenAIClient
from .rag import EmbeddingIndex


def unique_candidates_union(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge multiple SCAN outputs (self-consistency) into a unique candidate list by dopant_signature.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for rep in reports:
        for c in (rep.get("candidates") or []):
            sig = str(c.get("dopant_signature") or "").strip()
            if not sig:
                # make from elements
                els = [clean_element_symbol(x) for x in (c.get("dopant_elements") or [])]
                els = sorted({e for e in els if e and e not in HOST_SET})
                sig = "+".join(els)
                c["dopant_signature"] = sig
            if not sig:
                continue
            # merge: keep max confidence and concat evidence
            old = merged.get(sig)
            if old is None:
                merged[sig] = c
            else:
                old["confidence"] = max(float(old.get("confidence", 0.0)), float(c.get("confidence", 0.0)))
                old_e = old.get("evidence") or []
                new_e = c.get("evidence") or []
                old["evidence"] = (old_e + new_e)[:8]
                # prefer non-empty modifier_mode
                if str(old.get("modifier_mode","")).strip() in {"", "unknown"} and str(c.get("modifier_mode","")).strip():
                    old["modifier_mode"] = c.get("modifier_mode")
    # sort by confidence desc
    out = sorted(merged.values(), key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return out

def select_candidate_signatures(candidates: List[Dict[str, Any]], max_n: int = 8) -> List[str]:
    sigs = []
    for c in candidates:
        sig = str(c.get("dopant_signature") or "").strip()
        if sig:
            sigs.append(sig)
    # unique preserving order
    seen = set()
    out = []
    for s in sigs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:max_n]

def make_queries_for_candidate(sig: str) -> List[str]:
    # Use multiple queries to retrieve relevant chunks for this candidate.
    els = sig.split("+") if sig else []
    base = [
        f"{sig} NCM811 doped initial discharge capacity mAh/g",
        f"{sig} NCM811 doped precursor amount synthesis",
        f"{sig} co-doped NCM811 hydrothermal solvothermal",
        f"{sig} LiNi0.8Co0.1Mn0.1O2 doped performance",
    ]
    # add per-element queries to catch ionic notation
    for el in els:
        el = el.strip()
        if el:
            base.append(f"{el}3+ {el}2+ doped NCM811 initial discharge")
            base.append(f"{el} precursor nitrate sulfate acetate NCM811 doped")
    return base

def robust_get_top_chunks(index: EmbeddingIndex, client: OpenAIClient, queries: List[str], top_k_each: int = 6, cap_total: int = 18) -> List[Chunk]:
    # Retrieve top chunks for each query and union by chunk_id
    merged: Dict[str, Chunk] = {}
    for q in queries:
        for c in index.search(client, q, top_k=top_k_each):
            merged[c.chunk_id] = c
    # Return in deterministic order by doc_id then chunk index
    def key(c: Chunk):
        # chunk_id format doc::idx
        parts = c.chunk_id.split("::")
        idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (c.doc_id, idx)
    out = sorted(merged.values(), key=key)
    return out[:cap_total]
