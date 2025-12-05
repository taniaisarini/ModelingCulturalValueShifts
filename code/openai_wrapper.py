import asyncio
import random
from typing import List, Dict, Optional, Tuple
import openai
from openai import AsyncOpenAI
from openai.errors import RateLimitError, APIError, APIConnectionError, Timeout

class OpenAIAsyncParallel:
    def __init__(
        self,
        num_workers: int = 10,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        jitter: bool = True,
        timeout: int = 60,
        api_key: Optional[str] = None,
    ):
        """
        num_workers: max concurrent API calls
        max_retries: per-request retries
        base_backoff: initial backoff for errors
        max_backoff: cap for exponential backoff
        jitter: add randomness to avoid sync retries
        timeout: request timeout (seconds)
        """
        self.initial_workers = max(1, num_workers)
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
        self.timeout = timeout
        self.api_key = api_key

        self.client = AsyncOpenAI(api_key=api_key)

    async def _single_call(
        self, prompt: str, model: str, sem: asyncio.Semaphore
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Makes a single OpenAI call with retries and backoff.
        Returns a tuple: (response_text, top_logprobs_dict)
        """
        async with sem:
            attempt = 0
            while True:
                try:
                    resp = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1,  # first token only for logprobs
                            logprobs=5,    # top 5 token probabilities
                        ),
                        timeout=self.timeout,
                    )

                    choice = resp.choices[0]
                    text = choice.message.content
                    top_logprobs = choice.logprobs.top_logprobs[0] if choice.logprobs else None
                    return text, top_logprobs

                except (RateLimitError, APIConnectionError, Timeout, APIError) as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return f"Error after retries: {str(e)}", None

                    # exponential backoff with optional jitter
                    backoff = min(self.max_backoff, self.base_backoff * 2 ** (attempt - 1))
                    if self.jitter:
                        backoff *= (0.5 + random.random() / 2)
                    await asyncio.sleep(backoff)
                except Exception as e:
                    return f"Unexpected error: {str(e)}", None

    async def call_model(
        self,
        desired_workers: int,
        model: str,
        prompts: List[str],
    ) -> Dict[str, Tuple[str, Optional[Dict[str, float]]]]:
        """
        Main public API.
        - Splits work across workers.
        - Reduces concurrency if rate limits appear.
        - Returns a dict: {prompt: (response_text, top_logprobs_dict)}
        """
        current_workers = min(self.initial_workers, desired_workers)
        pending = {i: p for i, p in enumerate(prompts)}
        results: Dict[int, Tuple[str, Optional[Dict[str, float]]]] = {}

        while pending and current_workers >= 1:
            sem = asyncio.Semaphore(current_workers)
            tasks = {
                idx: asyncio.create_task(self._single_call(prompt, model, sem))
                for idx, prompt in pending.items()
            }

            finished = await asyncio.gather(*tasks.values())
            rate_limit_errors = 0
            new_pending = {}

            for (idx, prompt), result in zip(pending.items(), finished):
                text, top_logprobs = result
                if text.startswith("Error after retries"):
                    rate_limit_errors += 1
                    new_pending[idx] = prompt
                else:
                    results[idx] = (text, top_logprobs)

            # adaptive concurrency control
            if new_pending and rate_limit_errors > 0:
                new_workers = max(1, current_workers // 2)
                if new_workers < current_workers:
                    current_workers = new_workers
                else:
                    await asyncio.sleep(self.base_backoff)

            pending = new_pending

        # any remaining prompts failed
        for idx, p in pending.items():
            results[idx] = ("Failed after retries.", None)

        # convert index-mapped results to prompt-mapped results
        final = {prompts[i]: results[i] for i in range(len(prompts))}
        return final


# import asyncio

# client = OpenAIAsyncParallel(
#     num_workers=10,
#     max_retries=5,
#     api_key="sk-xxxx",
# )

# async def main():
#     prompts = [
#         "Choose A/B/C/D: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Rome",
#         "Choose A/B/C/D: Which animal is a mammal?\nA. Shark\nB. Dolphin\nC. Penguin\nD. Crocodile",
#     ]

#     results = await client.call_model(
#         desired_workers=5,
#         model="gpt-4o-mini",
#         prompts=prompts,
#     )

#     for prompt, (text, logprobs) in results.items():
#         print("\nPROMPT:", prompt)
#         print("RESPONSE:", text)
#         print("LOGPROBS:", logprobs)

# asyncio.run(main())
