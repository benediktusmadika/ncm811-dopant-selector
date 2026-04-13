# NCM811 Dopant Selector

(This is an ongoing research project. Please do not reuse or redistribute the code or methodology without proper attribution.)
  
It uses corpus-level RAG, OpenAI-based reasoning, deterministic chemistry calculations, and an optional closed-loop lab feedback cycle to recommend one actionable dopant recipe for NCM811 cathodes.

## What this project does

Given:

- a folder of PDFs about doped, co-doped, or coated NCM811 cathodes
- a `synthesis.csv` file containing your baseline hydrothermal or solvothermal synthesis conditions

the pipeline will:

1. parse the PDF corpus with GROBID first and local PDF extractors as fallback
2. build a corpus-wide embedding index
3. retrieve evidence across all papers together
4. identify dopant candidates with LLM-guided extraction
5. extract structured evidence for each candidate
6. rank candidates for **initial / first discharge capacity**
7. compute a deterministic weighing table in grams
8. generate a lab-ready synthesis protocol
9. optionally learn from previous lab trials through a feedback JSONL log
10. optionally export publication-style plots


## Project layout

```text
ncm811_dopant_selector_refactored/
├── dopant_search_labrecipe_rag_feedback.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── tests/
│   └── test_smoke.py
└── ncm811_dopant_selector/
    ├── __init__.py
    ├── __main__.py
    ├── chemistry.py
    ├── cli.py
    ├── constants.py
    ├── data_io.py
    ├── models.py
    ├── openai_client.py
    ├── optional_deps.py
    ├── pdf_ingestion.py
    ├── pipeline.py
    ├── plotting.py
    ├── prompts.py
    ├── rag.py
    ├── schemas.py
    ├── selection.py
    └── utils.py
```

## Module overview

- `cli.py`  
  Argument parsing and command-line entry points.

- `pipeline.py`  
  Main orchestration logic for the full workflow.

- `pdf_ingestion.py`  
  GROBID client, PDF extraction fallback logic, and chunking.

- `rag.py`  
  Embedding index build and vector retrieval.

- `openai_client.py`  
  OpenAI Responses API wrapper, structured output handling, JSON repair, schema coercion, and model capability handling.

- `schemas.py`  
  JSON schemas for scan, detail, decision, protocol, reflection, and mechanism-map stages.

- `prompts.py`  
  Prompt construction for each LLM stage.

- `chemistry.py`  
  Formula parsing, molar masses, stoichiometry, and weighing-table generation.

- `data_io.py`  
  `synthesis.csv` parsing and closed-loop feedback JSONL helpers.

- `plotting.py`  
  Ranking plot, sub-score plots, mismatch heatmap, mechanism map, and trajectory plotting.

- `selection.py`  
  Candidate merging, retrieval helpers, and query helpers.

- `utils.py`  
  Shared utilities such as logging, JSON writing, seed control, and element normalization.

## Requirements

Core dependencies:

```bash
pip install -r requirements.txt
```

The project expects Python 3.10 or newer.

### Optional but recommended

GROBID is recommended for better PDF parsing quality.

## Installation

### Option 1: run directly from the repository

```bash
pip install -r requirements.txt
```

Then set your OpenAI API key.

Linux or macOS:

```bash
export OPENAI_API_KEY="sk-..."
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

### Option 2: install as a package

```bash
pip install -e .
```

Then run:

```bash
ncm811-dopant-selector --help
```

## Quick start

### First recommendation

```bash
python dopant_search_labrecipe_rag_feedback.py   --pdf_dir pdf_files   --synthesis_csv synthesis.csv   --synthesis_row 0   --out results_iter1.json   --best_recipe_out best_recipe_iter1.json   --export_long_csv evidence_long_iter1.csv   --plots_dir plots_iter1   --plots_format pdf   --target_batch_mass_g 10   --li_excess_fraction 0.05   --seed 42   --verbose   --refresh_cache
```

### Add lab feedback and generate the next recommendation

```bash
python dopant_search_labrecipe_rag_feedback.py   --pdf_dir pdf_files   --synthesis_csv synthesis.csv   --synthesis_row 0   --feedback_path lab_feedback.jsonl   --add_feedback_from_recipe best_recipe_iter1.json   --measured_initial_discharge_mAh_g 214.7   --measured_c_rate 0.1   --measured_voltage_window "2.8-4.3 V"   --feedback_notes "Any deviations, atmosphere, loading, or handling notes"   --out results_iter2.json   --best_recipe_out best_recipe_iter2.json   --export_long_csv evidence_long_iter2.csv   --plots_dir plots_iter2   --plots_format pdf   --seed 42   --verbose
```

### Package entry point

You can also run the package directly:

```bash
python -m ncm811_dopant_selector --help
```

## Inputs

### PDF corpus

Put your papers in a directory such as:

```text
pdf_files/
```

### Synthesis CSV

The pipeline expects a `synthesis.csv` with baseline synthesis settings.  
It already contains defensive handling for several inconsistent calcination column names seen in real-world spreadsheets.

## Outputs

### `results.json`

The full pipeline result, including:

- metadata
- scan stage summary
- candidate details
- decision ranking
- mechanism map
- best candidate block
- protocol
- weighing table
- feedback context

### `best_recipe.json`

A smaller handoff file for the chosen recipe, including:

- dopant signature
- dopant precursors
- doping fraction and basis
- weighing table
- lab-ready protocol
- evidence summary
- inferred fields and warnings

### `evidence_long.csv`

A long-format export for downstream analysis.

### `plots_dir/`

When enabled, the pipeline can write:

- ranking plot
- sub-score bar plot
- radar plot
- mismatch heatmap
- mechanism-map heatmap
- closed-loop trajectory plot

### `lab_feedback.jsonl`

The closed-loop experiment history used for the next iteration.

## Key CLI options

### Core inputs

- `--pdf_dir`
- `--recursive`
- `--synthesis_csv`
- `--synthesis_row`

### OpenAI and model settings

- `--model`
- `--embedding_model`
- `--openai_api_key`
- `--temperature`
- `--reasoning_effort`
- `--seed`

### RAG and parsing

- `--grobid_url`
- `--no_grobid`
- `--grobid_cache_dir`
- `--rag_cache_dir`
- `--refresh_cache`
- `--chunk_chars`
- `--chunk_overlap_chars`
- `--top_k_scan`
- `--top_k_detail_each_query`
- `--cap_detail_chunks`
- `--self_consistency`

### Batch chemistry

- `--target_batch_mass_g`
- `--li_excess_fraction`

### Feedback loop

- `--feedback_path`
- `--no_feedback`
- `--add_feedback_from_recipe`
- `--measured_initial_discharge_mAh_g`
- `--measured_c_rate`
- `--measured_voltage_window`
- `--feedback_notes`
- `--feedback_dopant_signature`
- `--feedback_doping_fraction`
- `--feedback_doping_basis`
- `--feedback_modifier_mode`
- `--feedback_dopant_precursors`

### Output control

- `--out`
- `--best_recipe_out`
- `--export_long_csv`
- `--plots_dir`
- `--plots_format`
- `--debug_dir`
- `--verbose`

## Testing

Run the smoke tests with:

```bash
python -m unittest discover -s tests -v
```

The included test suite checks:

- formula parsing
- molar-mass calculation
- chunking guardrails
- synthesis CSV loading
- weighing-table generation
- CLI parsing
- candidate merge logic

## Validation status

Checked locally for this refactor:

- `python -m compileall` passed
- `python -m unittest discover -s tests -v` passed

Not checked locally in this environment:

- live OpenAI API calls
- full end-to-end execution against your private PDF corpus

That limitation is only because a real API key and the actual corpus are not available inside this environment.

## Notes on behavior

- host elements are constrained to `Li`, `Ni`, `Co`, `Mn`, and `O`
- any other element is treated as a dopant or modifier
- co-doping is supported through signatures like `Al+La`
- the weighing table is deterministic
- the LLM is used for candidate discovery, evidence synthesis, ranking, and protocol writing
- retrieval is corpus-level, not per-paper

## Troubleshooting

### No PDFs found

Check that `--pdf_dir` points to a directory containing `.pdf` files.

### GROBID not running

The pipeline automatically falls back to local PDF extractors, but quality may be lower.

### Missing OpenAI API key

Either pass `--openai_api_key` or set `OPENAI_API_KEY` in your environment.

### Invalid chunking configuration

Make sure:

```text
chunk_overlap_chars < chunk_chars
```

### Optional dependency missing

Install the full `requirements.txt` to avoid feature loss.

## Development notes

The refactor was intentionally conservative:

- it preserves the original pipeline stages
- it preserves the original prompt structure
- it preserves the original output artifacts
- it reduces coupling by separating concerns into modules

This makes future work easier, such as adding new ranking objectives, new extractors, or alternate protocol-generation strategies.
