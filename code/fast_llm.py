"""Fast, self-tuning batch LLM calling.

Sends many chat requests concurrently and adjusts how many are in flight on its own:
  * starts at `initial_concurrency` and ramps up while requests succeed
  * on a throttle (HTTP 429) it cuts concurrency sharply, pauses all new requests for the
    server's Retry-After (or a backoff), and retries the request
  * then probes upward again, slowly near the level that caused the throttle
Everything runs on one asyncio event loop (cheaper than hundreds of threads).

Caching (llm_cache.py): a call with the same model, messages and parameters is never sent twice. Each request is
looked up first; only misses go to the API, and every success is stored with its full response. That also makes
interrupted runs resumable. `plan_llm(...)` reports hits and misses without calling anything.

    from fast_llm import run_llm
    results = run_llm(
        [{"id": "q1", "system": "You are ...", "user": "Question ..."}, ...],
        model="gpt-4.1-mini", temperature=0.0, max_tokens=250)
    results["q1"]["text"]

Each request is {"id": str, "system": str, "user": str} or {"id": str, "messages": [...]}, plus an optional
"replicate": int to cache several independent samples of one prompt.
Each result is {"id", "text", "usage", "finish_reason", "attempts", "error", "cached", "cache_key"};
`error` is None on success, and the full stored response is in the cache under `cache_key`.
"""
import asyncio
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI,
                    DefaultAsyncHttpxClient, InternalServerError, RateLimitError)
from tqdm import tqdm

from llm_cache import DEFAULT_CACHE_DIR, LLMCache, make_key, new_record


class AdaptiveLimiter:
    """Concurrency limit that backs off on throttling and creeps back up on success (AIMD)."""

    def __init__(self, initial, minimum, maximum, decrease_factor=0.5, cooldown_s=2.0):
        self.limit = float(initial)
        self.minimum, self.maximum = minimum, maximum
        self.decrease_factor = decrease_factor
        self.cooldown_s = cooldown_s
        self.in_flight = 0
        self.throttles = 0
        self.ceiling = None          # concurrency level at the last throttle
        self._successes = 0
        self._last_decrease = 0.0
        self._pause_until = 0.0
        self._cond = None            # created inside the running loop

    def _condition(self):
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def acquire(self):
        while True:
            wait = self._pause_until - time.monotonic()
            if wait > 0:                       # a throttle told everyone to hold off
                await asyncio.sleep(wait)
                continue
            cond = self._condition()
            async with cond:
                await cond.wait_for(lambda: self.in_flight < int(self.limit))
                self.in_flight += 1
                return

    async def release(self):
        cond = self._condition()
        async with cond:
            self.in_flight -= 1
            cond.notify_all()

    def _wake(self):
        # limit changed; let waiters re-check (called from sync context inside the loop)
        asyncio.ensure_future(self._notify())

    async def _notify(self):
        cond = self._condition()
        async with cond:
            cond.notify_all()

    def on_success(self):
        self._successes += 1
        window = max(10, int(self.limit))
        if self._successes < window:
            return
        self._successes = 0
        if self.limit >= self.maximum:
            return
        near_ceiling = self.ceiling is not None and self.limit >= 0.8 * self.ceiling
        step = 1 if near_ceiling else max(1, int(0.1 * self.limit))
        self.limit = min(self.maximum, self.limit + step)
        self._wake()

    def on_throttle(self, retry_after=None):
        now = time.monotonic()
        self.throttles += 1
        if retry_after:
            self._pause_until = max(self._pause_until, now + min(retry_after, 60.0))
        if now - self._last_decrease >= self.cooldown_s:   # one cut per burst of 429s
            self.ceiling = self.limit
            self.limit = max(self.minimum, self.limit * self.decrease_factor)
            self._last_decrease = now
            self._successes = 0


def _retry_after(exc):
    try:
        v = exc.response.headers.get("retry-after")
        return float(v) if v is not None else None
    except Exception:
        return None


def _is_quota_error(exc):
    """A 429 that means 'out of money', not 'slow down'; retrying can never fix it."""
    body = getattr(exc, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    return code == "insufficient_quota"


def _messages(req):
    if "messages" in req:
        return req["messages"]
    msgs = []
    if req.get("system"):
        msgs.append({"role": "system", "content": req["system"]})
    msgs.append({"role": "user", "content": req["user"]})
    return msgs


def _raw(resp):
    """The complete API response as a plain dict (works across openai/pydantic versions)."""
    for name in ("model_dump", "dict"):
        if hasattr(resp, name):
            try:
                return json.loads(json.dumps(getattr(resp, name)(), default=str))
            except Exception:
                pass
    return {"repr": repr(resp)}


def _hit_result(rid, key, rec):
    r = rec["response"]
    return {"id": rid, "text": r["text"], "usage": r.get("usage_short", {}), "finish_reason": r.get("finish_reason"),
            "attempts": 0, "error": None, "cached": True, "cache_key": key}


def plan_llm(requests, model="gpt-4.1-mini", *, temperature=0.0, max_tokens=None, extra=None, cache_dir=None):
    """Which requests are already cached? Makes no LLM calls. Returns {"hits", "misses", "unique_misses"}
    (counts of requests; unique_misses counts distinct calls, since identical prompts are sent once)."""
    cache = LLMCache(cache_dir or DEFAULT_CACHE_DIR)
    params = _params(temperature, max_tokens, extra)
    hits, missing = 0, set()
    for req in requests:
        key = make_key(model, _messages(req), params, req.get("replicate", 0))
        if cache.has(model, key):
            hits += 1
        else:
            missing.add(key)
    n_miss = sum(1 for req in requests
                 if make_key(model, _messages(req), params, req.get("replicate", 0)) in missing)
    return {"requests": len(requests), "hits": hits, "misses": n_miss, "unique_misses": len(missing)}


def _params(temperature, max_tokens, extra):
    p = dict(extra or {})
    if temperature is not None:
        p["temperature"] = temperature
    if max_tokens is not None:
        p["max_tokens"] = max_tokens
    return p


async def run_llm_async(requests, model="gpt-4.1-mini", *, temperature=0.0, max_tokens=None,
                        extra=None, cache=True, cache_dir=None, initial_concurrency=64, min_concurrency=2,
                        max_concurrency=512, max_attempts=8, request_timeout=60.0, api_key=None,
                        client=None, progress=True):
    """Async version of run_llm (use this from your own event loop). Returns {id: result}.

    Every call is looked up in the on-disk cache (see llm_cache.py) by model + messages + parameters + the
    request's optional "replicate" number; only calls that are not cached are sent, and each success is stored
    with its full response. Identical requests inside one batch are sent once. Set cache=False to bypass it.
    """
    ids = [r["id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request ids must be unique")
    params = _params(temperature, max_tokens, extra)
    store = LLMCache(cache_dir or DEFAULT_CACHE_DIR) if cache else None

    results, todo = {}, {}          # todo: cache key -> [representative request, [ids sharing it]]
    for req in requests:
        msgs = _messages(req)
        key = make_key(model, msgs, params, req.get("replicate", 0))
        rec = store.get(model, key) if store else None
        if rec is not None:
            results[req["id"]] = _hit_result(req["id"], key, rec)
        elif key in todo:
            todo[key][1].append(req["id"])
        else:
            todo[key] = [req, [req["id"]]]
    if progress:
        print(f"{model}: {len(results)} of {len(requests)} requests found in cache; "
              f"sending {len(todo)} new call(s)")
    if not todo:
        return {i: results[i] for i in ids}

    own_client = client is None
    if own_client:
        http = DefaultAsyncHttpxClient(
            limits=httpx.Limits(max_connections=max_concurrency + 50,
                                max_keepalive_connections=max_concurrency),
            timeout=request_timeout)
        # max_retries=0: we handle every retry ourselves so throttles are visible to the controller
        client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             max_retries=0, http_client=http)

    limiter = AdaptiveLimiter(min(initial_concurrency, max_concurrency), min_concurrency, max_concurrency)
    kwargs = dict(params)
    bar = tqdm(total=len(todo), desc=f"{model}", unit="req", disable=not progress)
    stats = {"tokens": 0}

    def _tick():
        bar.set_postfix(conc=int(limiter.limit), inflight=limiter.in_flight,
                        throttles=limiter.throttles, tok=stats["tokens"], refresh=False)

    async def one(req, key):
        msgs, last_err = _messages(req), None
        for attempt in range(1, max_attempts + 1):
            await limiter.acquire()
            delay = None
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(model=model, messages=msgs, **kwargs)
                limiter.on_success()
                u = resp.usage
                short = {"prompt": u.prompt_tokens, "completion": u.completion_tokens} if u else {}
                stats["tokens"] += sum(short.values())
                choice = resp.choices[0]
                response = {"text": choice.message.content, "finish_reason": getattr(choice, "finish_reason", None),
                            "returned_model": getattr(resp, "model", None), "id": getattr(resp, "id", None),
                            "usage_short": short, "raw": _raw(resp)}
                if store:
                    store.put(new_record(key, model, msgs, params, req.get("replicate", 0), response,
                                         attempt, time.monotonic() - t0))
                return {"id": req["id"], "text": response["text"], "usage": short,
                        "finish_reason": response["finish_reason"], "attempts": attempt, "error": None,
                        "cached": False, "cache_key": key}
            except RateLimitError as e:
                if _is_quota_error(e):
                    return _fail(req, attempt, f"quota exhausted: {e}", key)
                last_err = f"429: {e}"
                ra = _retry_after(e)
                limiter.on_throttle(ra)
                delay = 0 if ra else min(30.0, 0.5 * 2 ** attempt)
            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                last_err = f"{type(e).__name__}: {e}"
                delay = min(30.0, 0.5 * 2 ** attempt)
            except APIStatusError as e:   # 4xx other than 429: the request itself is bad
                return _fail(req, attempt, f"{e.status_code}: {e}", key)
            finally:
                await limiter.release()
            await asyncio.sleep(delay * random.uniform(0.7, 1.3))
        return _fail(req, max_attempts, last_err, key)

    def _fail(req, attempts, err, key):
        return {"id": req["id"], "text": None, "usage": {}, "finish_reason": None, "attempts": attempts,
                "error": err, "cached": False, "cache_key": key}

    async def guarded(key, req, shared_ids):
        try:
            r = await one(req, key)
        except Exception as e:  # never let one request take down the batch
            r = _fail(req, 0, f"{type(e).__name__}: {e}", key)
        for rid in shared_ids:              # identical requests share one call and one answer
            results[rid] = dict(r, id=rid)
        bar.update(1)
        _tick()

    try:
        # feed tasks lazily so 100k requests do not create 100k live coroutines at once
        pending, it = set(), iter(todo.items())
        window = max_concurrency * 2
        while True:
            while len(pending) < window:
                item = next(it, None)
                if item is None:
                    break
                key, (req, shared_ids) = item
                pending.add(asyncio.ensure_future(guarded(key, req, shared_ids)))
            if not pending:
                break
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    finally:
        bar.close()
        if own_client:
            await client.close()

    failed = [i for i in ids if results[i].get("error")]
    if failed and progress:
        print(f"{len(failed)} of {len(ids)} requests failed after {max_attempts} attempts "
              f"(failures are not cached, so rerunning retries only those). First error: {results[failed[0]]['error']}")
    return {i: results[i] for i in ids}


def run_llm(requests, model="gpt-4.1-mini", **kwargs):
    """Run all requests and return {id: result}. Works in scripts and inside Jupyter."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_llm_async(requests, model, **kwargs))
    with ThreadPoolExecutor(1) as ex:   # already inside a running loop (e.g. a notebook)
        return ex.submit(lambda: asyncio.run(run_llm_async(requests, model, **kwargs))).result()
