"""
OpenAI API wrapper with batching and logprobs support.
"""

import openai
import os
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    from thread_gpt_suite.hybrid_gpt_mp_handler import HybridGPTMPHandler
    HYBRID_GPT_AVAILABLE = True
except ImportError:
    HYBRID_GPT_AVAILABLE = False
    logging.warning("hybrid_gpt_suite not available, multiprocessing will be disabled")

logger = logging.getLogger(__name__)


class OpenAIWrapper:
    """
    Wrapper for OpenAI API with support for:
    - Batch inference
    - Logprobs retrieval (for deferral decisions)
    - Multi-worker processing
    """
    
    def __init__(self, max_tokens=1500, temperature=0.1, reasoning_effort="low", 
                 num_processes=10, threads_per_process=60):
        # Get API key from environment (do NOT hardcode!)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.num_processes = num_processes
        self.threads_per_process = threads_per_process
        openai.api_key = self.api_key
        
        # Models that use max_completion_tokens instead of max_tokens
        self.new_param_models = ['o1-preview', 'o1-mini', 'gpt-4o', 'gpt-4o-mini']

    def infer(
        self, 
        prompts: List[str], 
        engine: str = 'gpt-4', 
        max_tokens: Optional[int] = None, 
        num_workers: int = 10, 
        system_prompt: str = 'You are a helpful assistant.',
        return_logprobs: bool = False
    ) -> List[str]:
        """
        Run inference on a batch of prompts.
        
        Args:
            prompts: List of prompts to process
            engine: OpenAI model name
            max_tokens: Maximum tokens to generate
            num_workers: Number of parallel workers
            system_prompt: System prompt to use
            return_logprobs: If True, return (responses, logprobs) tuple
            
        Returns:
            List of response strings, or (responses, logprobs) if return_logprobs=True
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        logger.info(f"Running inference on {len(prompts)} prompts with {num_workers} workers")
        
        # Check if model uses new parameter
        uses_new_param = any(model in engine for model in self.new_param_models)
        is_reasoning_model = 'gpt-5' in engine or 'o1' in engine
        
        # Hybrid multiprocessing + threading path (no logprobs support yet)
        if num_workers > 1 and HYBRID_GPT_AVAILABLE and not return_logprobs:
            # Build config - gpt_suite has been patched to use max_completion_tokens
            if uses_new_param:
                config = {
                    "temperature": self.temperature,
                    "max_completion_tokens": max_tokens
                }
            else:
                config = {
                    "temperature": self.temperature,
                    "max_tokens": max_tokens
                }
            
            # Add reasoning parameter for reasoning models
            if is_reasoning_model:
                config["reasoning"] = {"effort": self.reasoning_effort}
            
            handler = HybridGPTMPHandler(
                api_key=self.api_key, 
                gen_conf=config,
                num_processes=self.num_processes,
                threads_per_process=self.threads_per_process
            )
            batch = []
            for prompt in prompts:
                ins = {
                    'init_context': '',
                    'questions': [prompt],
                    'task_desc': system_prompt,
                    'model_name': engine
                }
                batch.append(ins)
            handler.add_batch(batch)
            outs = handler.process()
            
            # Handle empty responses
            responses = []
            for i, d in enumerate(outs):
                if d and len(d.values()) > 0:
                    responses.append(list(d.values())[0])
                else:
                    error_msg = f"Empty response from API for prompt {i+1}"
                    logger.error(error_msg)
                    responses.append(f"Error: {error_msg}")
            
            logger.info(f"Completed batch inference")
            return responses
        
        # Single-threaded path (with optional logprobs)
        else:
            if num_workers > 1 and not HYBRID_GPT_AVAILABLE:
                logger.warning("Multi-worker requested but hybrid_gpt_suite not available, using single-threaded")
            
            responses = []
            logprobs_list = []
            
            for i, prompt in enumerate(prompts):
                try:
                    logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                    
                    # Prepare parameters
                    params = {
                        "model": engine,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature,
                        "logprobs": return_logprobs,
                        "top_logprobs": 5 if return_logprobs else None
                    }
                    
                    # Use different API for gpt-5 reasoning models
                    if 'gpt-5' in engine:
                        # gpt-5 uses responses.create API with max_output_tokens
                        response = self.client.responses.create(
                            model=engine,
                            reasoning={"effort": self.reasoning_effort},
                            input=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_output_tokens=max_tokens,
                            temperature=self.temperature
                        )
                        content = response.output_text.strip()
                    else:
                        # Standard chat completions API for other models
                        # Use correct parameter name based on model
                        if any(model in engine for model in self.new_param_models):
                            params["max_completion_tokens"] = max_tokens
                        else:
                            params["max_tokens"] = max_tokens
                        
                        # Add reasoning parameter for o1 models (not gpt-5)
                        if 'o1' in engine:
                            params["reasoning"] = {"effort": self.reasoning_effort}
                        
                        response = self.client.chat.completions.create(**params)
                        content = response.choices[0].message.content.strip()
                    
                    responses.append(content)
                    
                    if return_logprobs:
                        # Extract logprobs from response
                        logprobs = response.choices[0].logprobs
                        logprobs_list.append(logprobs)
                    
                except Exception as e:
                    logger.error(f"Error processing prompt {i+1}: {e}")
                    responses.append(f'Error: {str(e)}')
                    if return_logprobs:
                        logprobs_list.append(None)
            
            logger.info(f"Completed single-threaded inference")
            
            if return_logprobs:
                return responses, logprobs_list
            return responses

    def infer_with_history(
        self, 
        prompts: List[List[str]], 
        engine: str = 'gpt-4', 
        max_tokens: Optional[int] = None, 
        num_workers: int = 1, 
        system_prompt: str = 'You are a helpful assistant.'
    ) -> List[List[str]]:
        """
        Run inference with conversation history.
        
        Args:
            prompts: List of conversation histories (each is a list of messages)
            engine: OpenAI model name
            max_tokens: Maximum tokens to generate
            num_workers: Number of parallel workers
            system_prompt: System prompt to use
            
        Returns:
            List of response histories
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        logger.info(f"Running inference with history on {len(prompts)} conversations")
        
        if num_workers > 1 and HYBRID_GPT_AVAILABLE:
            config = {"temperature": self.temperature, "max_tokens": max_tokens}
            handler = HybridGPTMPHandler(
                api_key=self.api_key, 
                gen_conf=config,
                num_processes=self.num_processes,
                threads_per_process=self.threads_per_process
            )
            batch = []
            for prompt in prompts:
                ins = {
                    'init_context': '',
                    'questions': prompt,
                    'task_desc': system_prompt,
                    'model_name': engine
                }
                batch.append(ins)
            handler.add_batch(batch)
            outs = handler.process()
            responses = [list(d.values()) for d in outs]
            return responses
        else:
            logger.warning("Single-threaded conversation history not fully implemented yet")
            # TODO: Implement single-threaded version with proper message history
            return [[]]


if __name__ == "__main__":
    # Test the wrapper
    wrapper = OpenAIWrapper()
    prompts = ["What is 2+2?", "What is the capital of France?"]
    results = wrapper.infer(prompts, num_workers=1)
    print("Results:", results)
