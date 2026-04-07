"""
OpenAI model implementation with caching and batch processing.
"""

from src.llm.model import Model
from src.llm.openai_wrapper import OpenAIWrapper
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    OpenAI model wrapper with caching.
    
    Features:
    - Automatic caching based on model configuration
    - Batch processing
    - Support for custom prompts from files
    """
    
    def __init__(
        self, 
        engine='gpt-4', 
        max_tokens=1500, 
        temperature=0.1, 
        batch_size=2000, 
        num_workers=10,
        reasoning_effort="low",
        num_processes=10,
        threads_per_process=60,
        **kwargs
    ):
        """
        Initialize the OpenAI model wrapper.

        Parameters:
            engine (str): The OpenAI engine/model to use.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            batch_size (int): How many prompts to process in a single batch.
            num_workers (int): Number of workers for parallel processing (deprecated).
            reasoning_effort (str): Reasoning effort for gpt-5/o1 models: low, medium, high.
            num_processes (int): Number of parallel processes for hybrid handler.
            threads_per_process (int): Number of threads per process for hybrid handler.
            **kwargs: Additional parameters if needed.
        """
        self.prompt_separated = True
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reasoning_effort = reasoning_effort
        self.num_processes = num_processes
        self.threads_per_process = threads_per_process
        self.wrapper = OpenAIWrapper(
            max_tokens=max_tokens, 
            temperature=temperature, 
            reasoning_effort=reasoning_effort,
            num_processes=num_processes,
            threads_per_process=threads_per_process
        )

        super().__init__()

    def _infer(self, data, **kwargs):
        """
        Runs inference on a dictionary containing prompts in batches.
        
        The dictionary should have:
            - 'prompts': a list of main prompt strings.
            - (optional) 'sys_prompt': a system prompt string.
        
        Returns:
            A list of responses.
        """
        prompts = data.get('prompts', [])
        sys_prompt = data.get('sys_prompt', 'You are a helpful assistant.')
        
        responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]
            batch_responses = self.wrapper.infer(
                batch_prompts,
                engine=self.engine,
                max_tokens=self.max_tokens,
                num_workers=self.num_workers,
                system_prompt=sys_prompt
            )
            responses.extend(batch_responses)
        return responses

    def load_prompt(self, prompt_name: str, system_prompt: bool = False) -> str:
        """
        Load a prompt from the prompts directory.
        
        Args:
            prompt_name: Name of the prompt (without extension)
            system_prompt: If True, load from sys_prompts/, else from main_prompts/
            
        Returns:
            Prompt text content
        """
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        
        if system_prompt:
            directory = project_root / "prompts" / "sys_prompts"
            filename = f"{prompt_name}_sys.txt"
        else:
            directory = project_root / "prompts" / "main_prompts"
            filename = f"{prompt_name}.txt"
        
        file_path = directory / filename
        
        if not file_path.exists():
            logger.warning(f"Prompt file not found: {file_path}")
            return ""
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        logger.debug(f"Loaded prompt from {file_path}")
        return content

    def get_cache_key(self):
        return {
            'engine': self.engine,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'reasoning_effort': self.reasoning_effort
        }
