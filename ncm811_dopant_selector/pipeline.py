from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chemistry import compute_weighing_table
from .constants import DEFAULT_SEED, HOST_ELEMENTS, LOGGER_NAME
from .data_io import (
    apply_overrides,
    build_feedback_record,
    load_feedback_jsonl,
    load_recipe_minimal,
    load_synthesis_csv,
    summarize_feedback_for_prompt,
    _jsonl_append,
)
from .models import Document, LabFeedback
from .openai_client import OpenAIClient
from .pdf_ingestion import DocumentLoader, chunk_text, discover_pdfs
from .plotting import (
    export_long_csv,
    plot_candidate_ranking,
    plot_closed_loop_trajectory,
    plot_mechanism_map_heatmap,
    plot_mismatch_heatmap,
    plot_subscores_bar,
    plot_subscores_radar,
)
from .prompts import (
    prompt_decide,
    prompt_decide_next,
    prompt_detail,
    prompt_mechanism_map,
    prompt_protocol,
    prompt_protocol_feedback,
    prompt_reflection_plan,
    prompt_scan,
)
from .rag import EmbeddingIndex
from .schemas import (
    schema_candidate_detail,
    schema_decision,
    schema_decision_next,
    schema_mechanism_map,
    schema_protocol,
    schema_reflection_plan,
    schema_scan_candidates,
)
from .selection import (
    make_queries_for_candidate,
    robust_get_top_chunks,
    select_candidate_signatures,
    unique_candidates_union,
)
from .utils import normalize_reasoning_effort, set_global_seed, setup_logging, write_json

LOG = logging.getLogger(LOGGER_NAME)


@dataclass
class PipelineConfig:
    pdf_dir: Path = Path("pdf_files")
    recursive: bool = False
    synthesis_csv: Path = Path("synthesis.csv")
    synthesis_row: int = 0

    model: str = "gpt-5.4-2026-03-05"
    embedding_model: str = "text-embedding-3-large"
    openai_api_key: Optional[str] = None
    seed: Optional[int] = DEFAULT_SEED
    temperature: Optional[float] = 0.2
    reasoning_effort: Optional[str] = None

    grobid_url: str = "http://localhost:8070"
    no_grobid: bool = False
    grobid_cache_dir: Optional[Path] = Path(".grobid_cache")

    rag_cache_dir: Path = Path(".rag_cache")
    refresh_cache: bool = False

    chunk_chars: int = 4000
    chunk_overlap_chars: int = 400
    top_k_scan: int = 18
    top_k_detail_each_query: int = 6
    cap_detail_chunks: int = 18
    self_consistency: int = 2

    target_batch_mass_g: float = 10.0
    li_excess_fraction: float = 0.05

    out: Path = Path("results.json")
    best_recipe_out: Path = Path("best_recipe.json")
    export_long_csv_path: Optional[Path] = Path("evidence_long.csv")
    plots_dir: Optional[Path] = None
    plots_format: str = "pdf"

    feedback_path: Optional[Path] = Path("lab_feedback.jsonl")
    no_feedback: bool = False
    add_feedback_from_recipe: Optional[Path] = None
    measured_initial_discharge_mAh_g: Optional[float] = None
    measured_c_rate: float = 0.1
    measured_voltage_window: Optional[str] = None
    feedback_notes: Optional[str] = None

    feedback_dopant_signature: Optional[str] = None
    feedback_doping_fraction: Optional[float] = None
    feedback_doping_basis: str = "unknown"
    feedback_modifier_mode: str = "unknown"
    feedback_dopant_precursors: Optional[str] = None

    debug_dir: Optional[Path] = None
    verbose: bool = False


def _append_feedback_if_requested(config: PipelineConfig, target) -> None:
    if config.no_feedback or config.feedback_path is None:
        return
    if config.measured_initial_discharge_mAh_g is None:
        return

    recipe_obj: Dict[str, Any] = {}
    recipe_source: Optional[str] = None

    if config.add_feedback_from_recipe:
        recipe_source = str(config.add_feedback_from_recipe)
        recipe_obj = load_recipe_minimal(config.add_feedback_from_recipe)
    elif config.feedback_dopant_signature:
        recipe_obj = {
            "dopant_signature": str(config.feedback_dopant_signature).strip(),
            "doping_fraction": config.feedback_doping_fraction,
            "doping_basis": str(config.feedback_doping_basis).strip() or "unknown",
            "modifier_mode": str(config.feedback_modifier_mode).strip() or "unknown",
            "dopant_precursors": [
                part.strip()
                for part in (config.feedback_dopant_precursors or "").split(",")
                if part.strip()
            ],
        }
    else:
        raise ValueError(
            "To append feedback, provide either --add_feedback_from_recipe <best_recipe.json> "
            "or --feedback_dopant_signature <sig>."
        )

    existing_trials = load_feedback_jsonl(config.feedback_path, synthesis_row_id=target.row_id)
    next_trial_id = max((trial.trial_id for trial in existing_trials), default=-1) + 1

    record = build_feedback_record(
        target=target,
        trial_id=next_trial_id,
        recipe=recipe_obj,
        measured_capacity_mAh_g=float(config.measured_initial_discharge_mAh_g),
        measured_c_rate=float(config.measured_c_rate),
        voltage_window=(str(config.measured_voltage_window).strip() if config.measured_voltage_window else None),
        notes=(str(config.feedback_notes).strip() if config.feedback_notes else None),
        recipe_path=recipe_source,
    )
    _jsonl_append(config.feedback_path, record)
    LOG.info("Appended feedback trial_id=%d to %s", next_trial_id, config.feedback_path)


def run_pipeline(config: PipelineConfig) -> int:
    setup_logging(config.verbose)
    set_global_seed(config.seed)
    config.reasoning_effort = normalize_reasoning_effort(config.reasoning_effort)
    LOG.info("Global seed configured: %s", config.seed)

    if not config.pdf_dir.exists():
        raise FileNotFoundError(config.pdf_dir)

    synthesis_rows = load_synthesis_csv(config.synthesis_csv)
    if config.synthesis_row < 0 or config.synthesis_row >= len(synthesis_rows):
        raise ValueError(f"--synthesis_row out of range: {config.synthesis_row} (rows={len(synthesis_rows)})")
    target = synthesis_rows[config.synthesis_row]
    LOG.info(
        "Target synthesis row %d loaded (method=%s, baseline_capacity=%s mAh/g).",
        target.row_id,
        target.method,
        target.baseline_capacity_mAh_g,
    )

    _append_feedback_if_requested(config, target)

    feedback_trials: List[LabFeedback] = []
    feedback_ctx: Dict[str, Any] = {
        "baseline_capacity_mAh_g": target.baseline_capacity_mAh_g,
        "trials": [],
        "summary": "no_feedback",
    }
    if not config.no_feedback and config.feedback_path is not None:
        feedback_trials = load_feedback_jsonl(config.feedback_path, synthesis_row_id=target.row_id)
        feedback_ctx = summarize_feedback_for_prompt(feedback_trials, target.baseline_capacity_mAh_g)
        if feedback_trials:
            LOG.info(
                "Loaded %d feedback trial(s) for synthesis_row=%d. %s",
                len(feedback_trials),
                target.row_id,
                feedback_ctx.get("summary"),
            )

    pdf_paths = discover_pdfs(config.pdf_dir, recursive=config.recursive)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {config.pdf_dir}")

    loader = DocumentLoader(
        use_grobid=not config.no_grobid,
        grobid_url=config.grobid_url,
        grobid_cache_dir=config.grobid_cache_dir,
    )

    documents: List[Document] = []
    LOG.info("Loading %d PDF(s)...", len(pdf_paths))
    for pdf_path in pdf_paths:
        document = loader.load_pdf(pdf_path)
        documents.append(document)
        from .pdf_ingestion import _alpha_score  # local import avoids exporting helper in package API

        LOG.info("Loaded %s | alpha=%.3f | chars=%d", document.doc_id, _alpha_score(document.text), len(document.text))

    chunks = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document,
                chunk_chars=config.chunk_chars,
                overlap_chars=config.chunk_overlap_chars,
            )
        )
    LOG.info("Chunked corpus into %d chunks.", len(chunks))

    client = OpenAIClient(
        api_key=config.openai_api_key,
        model=config.model,
        embedding_model=config.embedding_model,
        default_temperature=config.temperature,
        seed=config.seed,
    )

    index = EmbeddingIndex(cache_dir=config.rag_cache_dir, embedding_model=config.embedding_model)
    index.build(client, chunks, refresh=config.refresh_cache)

    reflection_plan: Optional[Dict[str, Any]] = None
    if feedback_trials:
        system_prompt, user_prompt = prompt_reflection_plan(target, feedback_ctx)
        debug_path = config.debug_dir / "reflect_plan.raw.txt" if config.debug_dir else None
        reflection_plan = client.call_json_schema(
            system=system_prompt,
            user=user_prompt,
            schema_name="reflection_plan",
            schema=schema_reflection_plan(),
            reasoning_effort=config.reasoning_effort,
            debug_path=debug_path,
        )
        LOG.info(
            "REFLECT/PLAN strategy=%s | avoid=%d | queries=%d",
            reflection_plan.get("exploration_strategy"),
            len(reflection_plan.get("avoid_exact_recipes") or []),
            len(reflection_plan.get("next_search_queries") or []),
        )

    if config.debug_dir:
        config.debug_dir.mkdir(parents=True, exist_ok=True)

    base_scan_queries = [
        "NCM811 doped co-doped initial discharge capacity first discharge capacity initial capacity 0.1C mAh/g dopant precursor amount",
        f"NCM811 {target.method} hydrothermal solvothermal doped precursor nitrate acetate sulfate pH calcination",
    ]
    extra_queries = []
    if reflection_plan and isinstance(reflection_plan.get("next_search_queries"), list):
        extra_queries = [str(q).strip() for q in reflection_plan.get("next_search_queries") or [] if str(q).strip()]

    scan_queries: List[str] = []
    seen_queries = set()
    for query in base_scan_queries + extra_queries:
        if query and query not in seen_queries:
            seen_queries.add(query)
            scan_queries.append(query)

    scan_chunks = robust_get_top_chunks(
        index,
        client,
        scan_queries,
        top_k_each=max(4, min(8, config.top_k_scan)),
        cap_total=config.top_k_scan,
    )
    scan_query = " | ".join(scan_queries)

    scan_reports = []
    for attempt in range(1, max(1, int(config.self_consistency)) + 1):
        system_prompt, user_prompt = prompt_scan(scan_chunks, target_method=target.method)
        debug_path = config.debug_dir / f"scan_attempt{attempt}.raw.txt" if config.debug_dir else None
        report = client.call_json_schema(
            system=system_prompt,
            user=user_prompt,
            schema_name="scan_candidates",
            schema=schema_scan_candidates(),
            reasoning_effort=config.reasoning_effort,
            debug_path=debug_path,
        )
        scan_reports.append(report)
        LOG.info("SCAN attempt %d/%d: got %d candidates.", attempt, max(1, int(config.self_consistency)), len(report.get("candidates") or []))

    merged_candidates = unique_candidates_union(scan_reports)
    if not merged_candidates:
        LOG.warning("SCAN found no candidates; using a conservative fallback candidate.")
        fallback_chunk_id = scan_chunks[0].chunk_id if scan_chunks else "unknown"
        merged_candidates = [
            {
                "dopant_elements": ["Al"],
                "dopant_signature": "Al",
                "modifier_mode": "bulk_TM_substitution",
                "evidence": [{"chunk_id": fallback_chunk_id, "quote": "fallback"}],
                "notes": "Fallback candidate inserted because scan returned empty.",
                "confidence": 0.1,
            }
        ]

    candidate_signatures = select_candidate_signatures(merged_candidates, max_n=8)
    LOG.info("Merged candidates (top %d): %s", len(candidate_signatures), candidate_signatures)

    candidate_details = []
    for signature in candidate_signatures:
        detail_chunks = robust_get_top_chunks(
            index,
            client,
            make_queries_for_candidate(signature),
            top_k_each=config.top_k_detail_each_query,
            cap_total=config.cap_detail_chunks,
        )
        system_prompt, user_prompt = prompt_detail(signature, target, detail_chunks)
        debug_path = config.debug_dir / f"detail_{signature}.raw.txt" if config.debug_dir else None
        detail = client.call_json_schema(
            system=system_prompt,
            user=user_prompt,
            schema_name="candidate_detail",
            schema=schema_candidate_detail(),
            reasoning_effort=config.reasoning_effort,
            debug_path=debug_path,
        )
        from .utils import clean_element_symbol

        elements = [clean_element_symbol(value) for value in (detail.get("dopant_elements") or [])]
        elements = sorted({value for value in elements if value and value not in HOST_ELEMENTS})
        detail["dopant_elements"] = elements
        detail["dopant_signature"] = "+".join(elements) if elements else detail.get("dopant_signature", signature)
        if not detail.get("dopant_signature"):
            detail["dopant_signature"] = signature

        inferred_fields = detail.get("inferred_fields") or []
        if detail.get("doping_fraction") is None:
            inferred_fields.append("doping_fraction_missing")
        if not detail.get("dopant_precursors"):
            inferred_fields.append("dopant_precursors_missing")
        detail["inferred_fields"] = list(dict.fromkeys(inferred_fields))

        candidate_details.append(detail)
        initial_capacity = detail.get("initial_discharge_capacity") or {}
        LOG.info(
            "DETAIL %s | trend=%s | doped=%s | baseline=%s | conf=%.2f",
            detail.get("dopant_signature"),
            initial_capacity.get("trend"),
            initial_capacity.get("doped_value_mAh_g"),
            initial_capacity.get("baseline_value_mAh_g"),
            float(detail.get("overall_confidence", 0.0)),
        )

    if feedback_trials:
        system_prompt, user_prompt = prompt_decide_next(target, candidate_details, feedback_ctx, reflection_plan)
        debug_path = config.debug_dir / "decide_next.raw.txt" if config.debug_dir else None
        decision = client.call_json_schema(
            system=system_prompt,
            user=user_prompt,
            schema_name="decision_next",
            schema=schema_decision_next(),
            reasoning_effort=config.reasoning_effort,
            debug_path=debug_path,
        )
    else:
        system_prompt, user_prompt = prompt_decide(target, candidate_details)
        debug_path = config.debug_dir / "decide.raw.txt" if config.debug_dir else None
        decision = client.call_json_schema(
            system=system_prompt,
            user=user_prompt,
            schema_name="decision",
            schema=schema_decision(),
            reasoning_effort=config.reasoning_effort,
            debug_path=debug_path,
        )

    best_signature = str(decision.get("best_dopant_signature") or "").strip()
    best_detail = next(
        (detail for detail in candidate_details if str(detail.get("dopant_signature") or "").strip() == best_signature),
        candidate_details[0],
    )
    if feedback_trials:
        best_detail = apply_overrides(best_detail, decision)

    LOG.info(
        "DECISION best=%s | doping_fraction=%s | basis=%s | confidence=%.2f",
        best_detail.get("dopant_signature"),
        best_detail.get("doping_fraction"),
        best_detail.get("doping_basis"),
        float(decision.get("confidence", 0.0)),
    )

    weighing_table = compute_weighing_table(
        target_mass_g=float(config.target_batch_mass_g),
        li_excess_fraction=float(config.li_excess_fraction),
        target=target,
        best_detail=best_detail,
    )

    if feedback_trials:
        system_prompt, user_prompt = prompt_protocol_feedback(target, best_detail, weighing_table, feedback_ctx, decision)
    else:
        system_prompt, user_prompt = prompt_protocol(target, best_detail, weighing_table)
    debug_path = config.debug_dir / "protocol.raw.txt" if config.debug_dir else None
    protocol = client.call_json_schema(
        system=system_prompt,
        user=user_prompt,
        schema_name="protocol",
        schema=schema_protocol(),
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=3000 if feedback_trials else 2800,
        debug_path=debug_path,
    )

    mechanism_map: Optional[Dict[str, Any]] = None
    if config.plots_dir:
        try:
            system_prompt, user_prompt = prompt_mechanism_map(target, decision.get("ranking") or [], candidate_details, top_k=8)
            debug_path = config.debug_dir / "mechanism_map.raw.txt" if config.debug_dir else None
            mechanism_map = client.call_json_schema(
                system=system_prompt,
                user=user_prompt,
                schema_name="mechanism_map",
                schema=schema_mechanism_map(),
                reasoning_effort=config.reasoning_effort,
                max_output_tokens=1800,
                temperature=0.2,
                debug_path=debug_path,
            )
            LOG.info(
                "MECHMAP: categories=%d | dopants=%d",
                len(mechanism_map.get("categories") or []),
                len(mechanism_map.get("dopants") or []),
            )
        except Exception as exc:
            LOG.warning("Mechanism map generation failed (continuing without it): %s", exc)

    out_payload = {
        "meta": {
            "script_version": "2026-04-refactor",
            "model": config.model,
            "embedding_model": config.embedding_model,
            "host_elements_constraint": list(HOST_ELEMENTS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_dir": str(config.pdf_dir),
            "synthesis_csv": str(config.synthesis_csv),
            "synthesis_row": target.row_id,
            "target_method": target.method,
            "target_batch_mass_g": config.target_batch_mass_g,
            "li_excess_fraction": config.li_excess_fraction,
            "seed": config.seed,
            "requested_temperature": config.temperature,
            "requested_reasoning_effort": config.reasoning_effort,
            "model_profile": client.describe_model_profile(),
            "iteration": (len(feedback_trials) + 1) if feedback_trials else 1,
            "feedback_path": str(config.feedback_path) if config.feedback_path else None,
            "feedback_trials_count": len(feedback_trials),
            "feedback_summary": feedback_ctx.get("summary"),
        },
        "feedback": feedback_ctx,
        "reflection_plan": reflection_plan,
        "scan": {
            "query": scan_query,
            "top_chunks_used": [chunk.chunk_id for chunk in scan_chunks],
            "merged_candidates": merged_candidates,
        },
        "candidate_details": candidate_details,
        "decision": decision,
        "mechanism_map": mechanism_map,
        "best": {
            "best_detail": best_detail,
            "weighing_table": weighing_table,
            "protocol": protocol,
        },
    }
    write_json(config.out, out_payload)
    LOG.info("Wrote results to %s", config.out)

    best_recipe = {
        "iteration": (len(feedback_trials) + 1) if feedback_trials else 1,
        "feedback_summary": feedback_ctx.get("summary"),
        "feedback_trials_used": feedback_ctx.get("trials"),
        "seed": config.seed,
        "requested_temperature": config.temperature,
        "requested_reasoning_effort": config.reasoning_effort,
        "model_profile": client.describe_model_profile(),
        "reflection_plan": reflection_plan,
        "decision": decision,
        "mechanism_map": mechanism_map,
        "dopant_signature": best_detail.get("dopant_signature"),
        "dopant_elements": best_detail.get("dopant_elements"),
        "dopant_precursors": best_detail.get("dopant_precursors"),
        "doping_level_text": best_detail.get("doping_level_text"),
        "doping_fraction": best_detail.get("doping_fraction"),
        "doping_basis": best_detail.get("doping_basis"),
        "baseline_synthesis_row": dataclasses.asdict(target),
        "weighing_table": weighing_table,
        "protocol": protocol,
        "evidence": {
            "initial_discharge_capacity": best_detail.get("initial_discharge_capacity"),
            "reported_synthesis": best_detail.get("reported_synthesis"),
        },
        "inferred_fields": best_detail.get("inferred_fields"),
        "warnings": best_detail.get("warnings"),
    }
    write_json(config.best_recipe_out, best_recipe)
    LOG.info("Wrote best recipe to %s", config.best_recipe_out)

    if config.export_long_csv_path:
        export_long_csv(candidate_details, config.export_long_csv_path)
        LOG.info("Wrote evidence CSV to %s", config.export_long_csv_path)

    if config.plots_dir:
        config.plots_dir.mkdir(parents=True, exist_ok=True)
        ranking = decision.get("ranking") or []

        ranking_path = config.plots_dir / f"ranking.{config.plots_format}"
        plot_candidate_ranking(ranking, ranking_path)
        LOG.info("Wrote ranking plot to %s", ranking_path)

        subscores_path = config.plots_dir / f"subscores_topk.{config.plots_format}"
        plot_subscores_bar(ranking, subscores_path, top_k=8)
        LOG.info("Wrote subscores plot to %s", subscores_path)

        best_row = next(
            (row for row in ranking if str(row.get("dopant_signature") or "").strip() == str(best_detail.get("dopant_signature") or "").strip()),
            ranking[0] if ranking else None,
        )
        if best_row:
            radar_path = config.plots_dir / f"radar_best.{config.plots_format}"
            plot_subscores_radar(best_row, radar_path)
            LOG.info("Wrote radar plot to %s", radar_path)

        mismatch_path = config.plots_dir / f"mismatch_heatmap.{config.plots_format}"
        mismatch_csv = config.plots_dir / "mismatch_table.csv"
        plot_mismatch_heatmap(ranking, candidate_details, target, mismatch_path, top_k=8, export_csv=mismatch_csv)
        LOG.info("Wrote mismatch heatmap to %s (csv=%s)", mismatch_path, mismatch_csv)

        if mechanism_map:
            mechanism_path = config.plots_dir / f"mechanism_map.{config.plots_format}"
            plot_mechanism_map_heatmap(mechanism_map, mechanism_path)
            LOG.info("Wrote mechanism map heatmap to %s", mechanism_path)
        else:
            LOG.info("Mechanism map not available; skipping mechanism plot.")

        if feedback_trials:
            trajectory_path = config.plots_dir / f"closed_loop_trajectory.{config.plots_format}"
            plot_closed_loop_trajectory(feedback_trials, trajectory_path, baseline_capacity=target.baseline_capacity_mAh_g)
            LOG.info("Wrote closed-loop trajectory plot to %s", trajectory_path)

    return 0
