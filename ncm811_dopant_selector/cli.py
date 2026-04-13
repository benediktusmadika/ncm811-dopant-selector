from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .constants import DEFAULT_SEED, SUPPORTED_REASONING_EFFORTS
from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NCM811 dopant selector (corpus-level RAG + GPT reasoning + lab-ready recipe + feedback loop)."
    )

    parser.add_argument("--pdf_dir", default="pdf_files")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--synthesis_csv", default="synthesis.csv")
    parser.add_argument("--synthesis_row", type=int, default=0)

    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--embedding_model", default="text-embedding-3-large")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning_effort",
        default="auto",
        choices=list(SUPPORTED_REASONING_EFFORTS),
        help="Reasoning effort to request when the model supports it. Use auto to omit the field.",
    )

    parser.add_argument("--grobid_url", default="http://localhost:8070")
    parser.add_argument("--no_grobid", action="store_true")
    parser.add_argument("--grobid_cache_dir", default=".grobid_cache")

    parser.add_argument("--rag_cache_dir", default=".rag_cache")
    parser.add_argument("--refresh_cache", action="store_true")

    parser.add_argument("--chunk_chars", type=int, default=4000)
    parser.add_argument("--chunk_overlap_chars", type=int, default=400)

    parser.add_argument("--top_k_scan", type=int, default=18)
    parser.add_argument("--top_k_detail_each_query", type=int, default=6)
    parser.add_argument("--cap_detail_chunks", type=int, default=18)
    parser.add_argument("--self_consistency", type=int, default=2)

    parser.add_argument("--target_batch_mass_g", type=float, default=10.0)
    parser.add_argument("--li_excess_fraction", type=float, default=0.05)

    parser.add_argument("--out", default="results.json")
    parser.add_argument("--best_recipe_out", default="best_recipe.json")
    parser.add_argument("--export_long_csv", default="evidence_long.csv")
    parser.add_argument("--plots_dir", default=None)
    parser.add_argument("--plots_format", default="pdf", choices=["pdf", "png"])

    parser.add_argument("--feedback_path", default="lab_feedback.jsonl")
    parser.add_argument("--no_feedback", action="store_true")
    parser.add_argument("--add_feedback_from_recipe", default=None)
    parser.add_argument("--measured_initial_discharge_mAh_g", type=float, default=None)
    parser.add_argument("--measured_c_rate", type=float, default=0.1)
    parser.add_argument("--measured_voltage_window", default=None)
    parser.add_argument("--feedback_notes", default=None)

    parser.add_argument("--feedback_dopant_signature", default=None)
    parser.add_argument("--feedback_doping_fraction", type=float, default=None)
    parser.add_argument("--feedback_doping_basis", default="unknown")
    parser.add_argument("--feedback_modifier_mode", default="unknown")
    parser.add_argument("--feedback_dopant_precursors", default=None)

    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _maybe_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    args = build_parser().parse_args(argv)
    return PipelineConfig(
        pdf_dir=Path(args.pdf_dir),
        recursive=bool(args.recursive),
        synthesis_csv=Path(args.synthesis_csv),
        synthesis_row=int(args.synthesis_row),
        model=str(args.model),
        embedding_model=str(args.embedding_model),
        openai_api_key=args.openai_api_key,
        seed=args.seed,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        grobid_url=str(args.grobid_url),
        no_grobid=bool(args.no_grobid),
        grobid_cache_dir=_maybe_path(args.grobid_cache_dir),
        rag_cache_dir=Path(args.rag_cache_dir),
        refresh_cache=bool(args.refresh_cache),
        chunk_chars=int(args.chunk_chars),
        chunk_overlap_chars=int(args.chunk_overlap_chars),
        top_k_scan=int(args.top_k_scan),
        top_k_detail_each_query=int(args.top_k_detail_each_query),
        cap_detail_chunks=int(args.cap_detail_chunks),
        self_consistency=int(args.self_consistency),
        target_batch_mass_g=float(args.target_batch_mass_g),
        li_excess_fraction=float(args.li_excess_fraction),
        out=Path(args.out),
        best_recipe_out=Path(args.best_recipe_out),
        export_long_csv_path=_maybe_path(args.export_long_csv),
        plots_dir=_maybe_path(args.plots_dir),
        plots_format=str(args.plots_format),
        feedback_path=_maybe_path(args.feedback_path),
        no_feedback=bool(args.no_feedback),
        add_feedback_from_recipe=_maybe_path(args.add_feedback_from_recipe),
        measured_initial_discharge_mAh_g=args.measured_initial_discharge_mAh_g,
        measured_c_rate=float(args.measured_c_rate),
        measured_voltage_window=args.measured_voltage_window,
        feedback_notes=args.feedback_notes,
        feedback_dopant_signature=args.feedback_dopant_signature,
        feedback_doping_fraction=args.feedback_doping_fraction,
        feedback_doping_basis=str(args.feedback_doping_basis),
        feedback_modifier_mode=str(args.feedback_modifier_mode),
        feedback_dopant_precursors=args.feedback_dopant_precursors,
        debug_dir=_maybe_path(args.debug_dir),
        verbose=bool(args.verbose),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_pipeline(parse_args(argv))
