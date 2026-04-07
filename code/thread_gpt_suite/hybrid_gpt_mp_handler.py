"""
Hybrid multiprocessing + threading handler for maximum throughput.

This implementation uses multiple processes (to utilize multiple CPUs) where each
process spawns multiple threads (to handle concurrent API calls). This maximizes
both CPU utilization and network I/O concurrency.

Example:
    - 8 processes × 50 threads each = 400 concurrent API calls
    - Uses 8 CPUs for parallel processing
    - Each CPU handles 50 concurrent network requests
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict
from warnings import warn
from os import cpu_count, environ
import copy
import inspect
import logging
import multiprocessing as mp
from . import thread_gpt_util

_logger = logging.getLogger(__name__)


def _process_chunk_with_threads(args):
    """
    Process a chunk of items using threading within a single process.
    This function is called by each worker process.
    """
    chunk, api_key, num_threads, openai_args = args
    
    # Initialize thread-local client for this process
    thread_gpt_util.initialize(api_key=api_key, **openai_args)
    
    # Use threading to process items in this chunk
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(thread_gpt_util.generate_explanation_wrapper, item): i 
            for i, item in enumerate(chunk)
        }
        
        # Collect results in order
        temp_results = [None] * len(chunk)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                temp_results[idx] = future.result()
            except Exception as e:
                _logger.error(f"Error processing item {idx}: {e}")
                temp_results[idx] = {}
        
        results = temp_results
    
    return results


class HybridGPTMPHandler:
    """
    Hybrid handler using both multiprocessing and threading.
    
    This maximizes throughput by:
    1. Using multiple processes to utilize all available CPUs
    2. Each process spawns multiple threads for concurrent API calls
    3. Total concurrency = num_processes × threads_per_process
    
    For example, with 8 processes and 50 threads each:
    - Uses 8 CPUs in parallel
    - Each CPU handles 50 concurrent API requests
    - Total of 400 concurrent API calls
    """
    
    def __init__(self, 
                 api_key: str = environ.get('OPENAI_API_KEY'),
                 num_processes: int = None,
                 threads_per_process: int = 50,
                 gen_conf: dict = None,
                 max_retries: int = 2,
                 **kwargs):
        """
        Initialize the Hybrid GPT Handler.
        
        Args:
            api_key: OpenAI API Key
            num_processes: Number of worker processes (default: CPU count)
            threads_per_process: Number of threads per process (default: 50)
            gen_conf: Generation parameters
            max_retries: Number of retries for failed items
            **kwargs: Additional OpenAI client arguments
        """
        assert api_key, "OpenAI API Key is required"
        
        self.api_key = api_key
        self.openai_args = kwargs
        self.gen_conf = gen_conf if gen_conf else {}
        
        # Default to CPU count for processes
        if num_processes is None:
            num_processes = min(cpu_count(), 16)  # Cap at 16 to avoid overwhelming
        
        self.num_processes = num_processes
        self.threads_per_process = threads_per_process
        self.total_workers = num_processes * threads_per_process
        
        self.queue: List[dict] = []
        self.max_retries = max_retries
        
        # Generate function signature mappings
        func_signature = inspect.signature(thread_gpt_util.generate_explanation)
        func_arg_list = list(func_signature.parameters.keys())
        if 'kwargs' in func_arg_list:
            func_arg_list.remove('kwargs')
        self.func_required_arg_list = [
            arg for arg in func_arg_list 
            if func_signature.parameters[arg].default is inspect.Parameter.empty
        ]
        self.func_args = {
            arg: func_signature.parameters[arg].annotation 
            for arg in func_arg_list
        }
        
        _logger.info(
            f"Initialized HybridGPTMPHandler: "
            f"{num_processes} processes × {threads_per_process} threads = "
            f"{self.total_workers} total concurrent workers"
        )
    
    def _dict_verifier(self, input_dict: dict) -> bool:
        """Verify if the input dictionary is valid for processing"""
        for k in self.func_required_arg_list:
            if k not in input_dict:
                _logger.error(f"Invalid dictionary, key {k} is required")
                return False
        for k, v in input_dict.items():
            if k not in self.func_args:
                continue
            if not isinstance(v, self.func_args[k]):
                _logger.error(f"Invalid dictionary, item {k} is not of type {self.func_args[k]}")
                return False
        return True
    
    def add_batch(self, batch: List[dict]):
        """
        Add a batch of instances to the queue.
        
        Args:
            batch: List of dictionaries with required arguments
        """
        # Quick validation using threading (faster than multiprocessing for small checks)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(batch), 100)) as executor:
            verification_results = list(executor.map(self._dict_verifier, batch))
        
        if False in verification_results:
            raise AssertionError("Verification failed for some items")
        
        for b in batch:
            b['openai_args'] = {'api_key': self.api_key}
            b['openai_args'].update(self.openai_args)
            b.update(self.gen_conf)
            if 'max_retries' not in b['openai_args']:
                b['openai_args']['max_retries'] = 5
            self.queue.append(copy.deepcopy(b))
        
        _logger.info(f"Added {len(batch)} items to queue. Total queue size: {len(self.queue)}")
    
    def process(self, rerun_on_error: bool = False) -> List[Dict[str, str]]:
        """
        Process the batch using hybrid multiprocessing + threading.
        
        Args:
            rerun_on_error: Whether to retry failed items
            
        Returns:
            List of result dictionaries
        """
        queue = self.queue[:]
        self.queue = []
        
        if not queue:
            return []
        
        _logger.info(
            f"Processing {len(queue)} items with {self.num_processes} processes "
            f"× {self.threads_per_process} threads/process = {self.total_workers} total workers"
        )
        
        # Split work into chunks for each process
        chunk_size = max(1, len(queue) // self.num_processes)
        chunks = [
            queue[i:i + chunk_size] 
            for i in range(0, len(queue), chunk_size)
        ]
        
        # Prepare arguments for each process
        process_args = [
            (chunk, self.api_key, self.threads_per_process, self.openai_args)
            for chunk in chunks
        ]
        
        # Process chunks with multiprocessing
        results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            chunk_results = list(executor.map(_process_chunk_with_threads, process_args))
            
            # Flatten results
            for chunk_result in chunk_results:
                results.extend(chunk_result)
        
        # Handle retries if requested
        if rerun_on_error:
            failed_indexes = [i for i, r in enumerate(results) if len(r) == 0]
            
            for retry in range(self.max_retries):
                if not failed_indexes:
                    break
                
                _logger.warning(
                    f"Retry {retry + 1}/{self.max_retries}: "
                    f"{len(failed_indexes)} failed items"
                )
                
                # Re-queue failed items
                for idx in failed_indexes:
                    self.queue.append(queue[idx])
                
                # Retry
                retry_results = self.process(rerun_on_error=False)
                
                # Update results
                new_failed = []
                for i, idx in enumerate(failed_indexes):
                    if len(retry_results[i]) > 0:
                        results[idx] = retry_results[i]
                    else:
                        new_failed.append(idx)
                
                failed_indexes = new_failed
        
        _logger.info(f"Processing completed: {sum(1 for r in results if len(r) > 0)}/{len(results)} successful")
        return results
