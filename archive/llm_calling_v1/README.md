# Archived: old LLM calling code

Archived 2026-09-24. Replaced by `code/fast_llm.py`, which is the only LLM caller to use going forward.

- `thread_gpt_suite/` - 200-350 thread pool around synchronous OpenAI calls. Fixed concurrency, no throttle
  handling beyond SDK retries, no checkpointing, default temperature 0.9. About 5.7x slower than `fast_llm` on a
  50-request benchmark (9.2 s vs 1.6 s). Always sends `max_completion_tokens`, so it fails on older `openai`
  packages (it only works in the `wvs` conda env).
- `llm/` - model wrapper classes built on `thread_gpt_suite`; nothing else used them.

`code/parse_codebok.ipynb` still imports `thread_gpt_suite` (its output, `data/parsed_survey_data*.jsonl`, was
already produced). To rerun it, switch that notebook to `fast_llm.run_llm`.
