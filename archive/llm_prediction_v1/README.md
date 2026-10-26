# Archived: first round of LLM prediction experiments

Archived 2026-09-24 because the results are not trusted and are being redone from scratch with a
cleaner setup. Nothing here has been deleted; paths inside the code and notebooks still point at the old
locations, so they will not run as-is.

## What is here

- `code/run_survey_on_llm.py` - direct baseline: ask the LLM how a country's average answer shifted between two waves.
- `code/persona_based_prediction_llm_runner.py` - persona simulation: one prompt per real respondent, 10 questions per prompt.
- `code/dataloader.py` - older Welch-only data loader, only used by the direct baseline (superseded by `code/WVS_dataloader.py`).
- `code/*.ipynb` - notebooks that ran or explored these pipelines (`run_llm_baseline`, `run_llm_basline_changes`,
  `persona_based_prediction` and its copy, `CoT_prompt_prediction`, `starter`).
- `prompts/` - system and user prompts for the direct baseline, persona-based and chain-of-thought variants.
- `results/` - raw and parsed outputs. Direct baseline: USA and IND, wave 6. Persona-based (`results_new/`):
  USA, EGY, TUR, UKR, waves 5-7 (plus a duplicate `UKR_6_` run).

## Known problems found before archiving

- The multi-select child-qualities question was never answered by the LLM (all 21k answers empty): the parser
  looked for column codes (A029, ...) in the reply, but the prompt showed the options as letters A-L.
- Persona predictions matched real answers 57% of the time versus 69% for always predicting the most common
  answer, and predicted wave-to-wave shifts did not track real shifts (correlation about 0.05).
- The CoT user prompt has a typo (`{year 2}`).

## Not archived (still used)

`code/thread_gpt_suite/` and `code/llm/` (LLM-calling plumbing), `code/WVS_dataloader.py`, and the codebook
parsing work (`code/parse_codebok.ipynb`, `code/prompts/*/codebook_parsing/create_survey.txt`, `data/parsed_survey_data*.jsonl`).
