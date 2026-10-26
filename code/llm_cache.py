"""On-disk cache of LLM calls: the same (model, prompt, parameters) is never sent twice.

A call is identified by a hash of everything that can change the answer:
    model name, the full message list (system + user), the generation parameters (temperature, max_tokens,
    top_p, seed, ...), and a `replicate` number (use 0, 1, 2... to deliberately cache several independent
    samples of the same prompt).
Only successful calls are stored. Each stored record keeps the full response (text, finish reason, token
usage, and the complete raw API response) together with the exact request that produced it.

Layout (append-only JSONL, chosen over SQLite because this project lives on NFS where SQLite locking is unsafe):
    llm_cache/<model>/<first two hex chars of key>.jsonl
Shards are read lazily and only when a key in them is looked up. If a key ever appears twice, the first
record wins. Truncated or corrupt lines (e.g. from a crash mid-write) are skipped.
"""
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "llm_cache"


def clean_params(params):
    """Generation parameters that identify a call; None values mean 'not set' and are dropped."""
    return {k: v for k, v in (params or {}).items() if v is not None}


def make_key(model, messages, params=None, replicate=0):
    payload = {"model": model, "messages": messages, "params": clean_params(params), "replicate": replicate}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class LLMCache:
    def __init__(self, root=DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self._shards = {}

    def _path(self, model, key):
        return self.root / _safe(model) / f"{key[:2]}.jsonl"

    def _shard(self, model, key):
        path = self._path(model, key)
        if path not in self._shards:
            records = {}
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        records.setdefault(rec["key"], rec)
            self._shards[path] = records
        return self._shards[path]

    def get(self, model, key):
        """The stored record for this key, or None."""
        return self._shard(model, key).get(key)

    def has(self, model, key):
        return self.get(model, key) is not None

    def put(self, record):
        """Append a record (must have 'key' and 'model'). A key already present is left untouched."""
        model, key = record["model"], record["key"]
        shard = self._shard(model, key)
        if key in shard:
            return
        path = self._path(model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)  # one write per record keeps lines intact even if the run is interrupted
        shard[key] = record

    def records(self, model=None):
        """Iterate over every stored record (optionally only one model). Reads the files directly."""
        dirs = [self.root / _safe(model)] if model else sorted(p for p in self.root.glob("*") if p.is_dir())
        for d in dirs:
            for path in sorted(d.glob("*.jsonl")):
                seen = set()
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec["key"] not in seen:
                            seen.add(rec["key"])
                            yield rec

    def stats(self):
        """{model: number of cached calls}."""
        counts = {}
        for rec in self.records():
            counts[rec["model"]] = counts.get(rec["model"], 0) + 1
        return counts


def new_record(key, model, messages, params, replicate, response, attempts, latency_s):
    """Assemble a cache record. `response` is a dict with text, finish_reason, returned_model, id, usage, raw."""
    return {"key": key, "model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request": {"messages": messages, "params": clean_params(params), "replicate": replicate},
            "response": response, "attempts": attempts, "latency_s": round(latency_s, 3)}
