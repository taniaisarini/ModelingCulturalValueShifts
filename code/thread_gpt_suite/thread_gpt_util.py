"""
Thread-based GPT utility for high-throughput API interactions.
Uses threading/asyncio to maximize rate limit utilization.
Interface-compatible with gpt_suite.gpt_util.
"""

import os
import re
import json
import logging
import asyncio
from pprint import pprint
from datetime import datetime
from typing import Optional, Dict, Union, List
from warnings import warn
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

# Housekeeping Variables
_generation_config = {"temperature": 0.9,
                      "max_completion_tokens": 4096,
                      "top_p": 1.0,
                      "frequency_penalty": 0.0,
                      "presence_penalty": 0.0}
_reasoning_config = {"effort": "low"}  # For reasoning models like gpt-5
_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None
_default_model: str = "gpt-3.5-turbo" \
    if os.environ.get('GU_DEFAULT_MODEL') is None \
    else os.environ.get('GU_DEFAULT_MODEL')
_target_majors = [1]
_image_special_token = "{IMAGE_PLH}"
_logger = logging.getLogger(__name__)

# Thread-local storage for clients
_thread_local = threading.local()


def initialize(api_key: str = None, **kwargs):
    """
    Initialize the OpenAI client with API key.
    Thread-safe initialization for use with multiprocessing.
    
    :param api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    :param kwargs: Additional arguments for OpenAI client
    """
    global _client, _async_client
    
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
    
    _client = OpenAI(api_key=api_key, **kwargs)
    _async_client = AsyncOpenAI(api_key=api_key, **kwargs)
    _logger.info("Thread-based OpenAI client initialized")


def _version_checker(target_majors: list) -> bool:
    """
    Check if the current version of OpenAI is compatible with the target major versions
    :param target_majors: List of target major versions
    :type target_majors: list

    :return: True if OpenAI version is larger or equal to the minimal version, False otherwise
    :rtype: bool
    """
    minimal_version = min(target_majors)
    # Check if compatible with openai package
    major = int(openai.__version__.split('.')[0])
    if major not in target_majors:
        _logger.warning(f"OpenAI({major}.x) has not been tested on this version yet. Please use with caution.")
        warn(f"OpenAI({major}.x) has not been tested on this version yet. Please use with caution.")
    return int(major) >= minimal_version


def _get_thread_client(api_key: str, **kwargs) -> OpenAI:
    """
    Get or create a thread-local OpenAI client.
    Each thread gets its own client to avoid conflicts.
    
    :param api_key: OpenAI API key
    :param kwargs: Additional arguments for OpenAI client
    :return: Thread-local OpenAI client
    """
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = OpenAI(api_key=api_key, **kwargs)
    return _thread_local.client


async def _get_response_async(message: list,
                               model_name: str = _default_model,
                               debug_log_path: str = None,
                               client: AsyncOpenAI = None,
                               **kwargs) -> ChatCompletion:
    """
    Async wrapper function to get response from OpenAI
    :param message: List of messages in the format defined in OpenAI API
    :type message: list

    :param model_name: Name for the OpenAI model to use, defaults to _default_model
    :type model_name: str(, optional)

    :param debug_log_path: Path to save the debug log, defaults to None
    :type debug_log_path: str(, optional)

    :param client: AsyncOpenAI client object
    :type client: AsyncOpenAI

    :param kwargs: Generation parameters
    :type kwargs: dict(, optional)

    :return: ChatCompletion object from OpenAI
    :rtype: ChatCompletion
    """
    # Check is the utility is properly setup
    assert client is not None, "need to provide async client."
    
    # Prepare generation arguments
    args = {}
    for k in _generation_config:
        args[k] = kwargs[k] if k in kwargs else _generation_config[k]
    
    # Get response from OpenAI
    _logger.debug(f"Getting async response from OpenAI for model: {model_name}")
    
    # Use different API for gpt-5 reasoning models
    if 'gpt-5' in model_name:
        reasoning = kwargs.get('reasoning', _reasoning_config)
        input_messages = []
        for msg in message:
            input_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Build params for responses.create (no temperature for gpt-5)
        response_params = {
            "model": model_name,
            "reasoning": reasoning,
            "input": input_messages,
            "max_output_tokens": args.get('max_completion_tokens', 4096)
        }
        
        result = await client.responses.create(**response_params)
        
        # Convert response format
        if hasattr(result, 'output_text'):
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content, 'role': 'assistant'})()
                    self.finish_reason = 'stop'
            
            class MockCompletion:
                def __init__(self, output_text):
                    self.choices = [MockChoice(output_text)]
                    self.model = model_name
                    self.usage = None
                
                def model_dump(self, **kwargs):
                    return {
                        'choices': [{'message': {'content': self.choices[0].message.content, 'role': 'assistant'}}],
                        'model': self.model
                    }
            
            result = MockCompletion(result.output_text)
    else:
        # Standard chat completions API for other models
        result = await client.chat.completions.create(
            model=model_name,
            messages=message,
            **args
        )
    
    _logger.debug(f"Async response received from OpenAI")
    
    # Save output if debug path was given
    if debug_log_path is not None:
        _logger.debug(f"Saving raw output to: {debug_log_path}")
        os.makedirs(debug_log_path, exist_ok=True)
        output_file_name = datetime.now().strftime(model_name + "_log_%Y_%m_%d_%H_%M_%S_%f.json")
        with open(os.path.join(debug_log_path, output_file_name), "w") as fp:
            result_dict = result.model_dump(exclude_unset=True)
            result_dict['message'] = message
            fp.write(json.dumps(result_dict, indent=2))
    
    return result


def _get_response(message: list,
                  model_name: str = _default_model,
                  debug_log_path: str = None,
                  client: OpenAI = None,
                  **kwargs) -> ChatCompletion:
    """
    Wrapper function to get response from OpenAI (synchronous version for compatibility)
    :param message: List of messages in the format defined in OpenAI API
    :type message: list

    :param model_name: Name for the OpenAI model to use, defaults to _default_model
    :type model_name: str(, optional)

    :param debug_log_path: Path to save the debug log, defaults to None
    :type debug_log_path: str(, optional)

    :param client: OpenAI client object, defaults to `_client` if not provided
    :type client: OpenAI(, optional)

    :param kwargs: Generation parameters
    :type kwargs: dict(, optional)

    :return: ChatCompletion object from OpenAI
    :rtype: ChatCompletion
    """
    # Use global client if none provided
    if client is None:
        client = _client
    
    # Check is the utility is properly setup
    assert client is not None, "need to run init() first or provide client."
    
    # Prepare generation arguments
    args = {}
    for k in _generation_config:
        args[k] = kwargs[k] if k in kwargs else _generation_config[k]
    
    # Get response from OpenAI
    _logger.debug(f"Getting response from OpenAI for model: {model_name}")
    
    # Use different API for gpt-5 reasoning models
    if 'gpt-5' in model_name:
        reasoning = kwargs.get('reasoning', _reasoning_config)
        input_messages = []
        for msg in message:
            input_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Build params for responses.create (no temperature for gpt-5)
        response_params = {
            "model": model_name,
            "reasoning": reasoning,
            "input": input_messages,
            "max_output_tokens": args.get('max_completion_tokens', 4096)
        }
        
        result = client.responses.create(**response_params)
        
        # Convert response format
        if hasattr(result, 'output_text'):
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content, 'role': 'assistant'})()
                    self.finish_reason = 'stop'
            
            class MockCompletion:
                def __init__(self, output_text):
                    self.choices = [MockChoice(output_text)]
                    self.model = model_name
                    self.usage = None
                
                def model_dump(self, **kwargs):
                    return {
                        'choices': [{'message': {'content': self.choices[0].message.content, 'role': 'assistant'}}],
                        'model': self.model
                    }
            
            result = MockCompletion(result.output_text)
    else:
        # Standard chat completions API for other models
        result = client.chat.completions.create(
            model=model_name,
            messages=message,
            **args
        )
    
    _logger.debug(f"Response received from OpenAI")
    
    # Save output if debug path was given
    if debug_log_path is not None:
        _logger.debug(f"Saving raw output to: {debug_log_path}")
        os.makedirs(debug_log_path, exist_ok=True)
        output_file_name = datetime.now().strftime(model_name + "_log_%Y_%m_%d_%H_%M_%S_%f.json")
        with open(os.path.join(debug_log_path, output_file_name), "w") as fp:
            result_dict = result.model_dump(exclude_unset=True)
            result_dict['message'] = message
            fp.write(json.dumps(result_dict, indent=2))
    
    return result


def _image_verification(image_url: str) -> bool:
    """
    Verify if the image provided has a valid format
    :param image_url: image in base64 format or a URL
    :type image_url: str

    :return: True if the image_url is valid, False otherwise
    :rtype: bool
    """
    if image_url.startswith("data:image/jpeg;base64,"):
        return True
    if re.match(r'^https?://', image_url):
        return True
    return False


def _vision_question_verification(question: Dict[str, Union[str, List[str]]]) -> bool:
    """
    Verify if the dictionary provided has a valid format
    :param question: Dictionary with 'text' and 'image' keys
    :type question: Dict[str, Union[str, List[str]]]

    :return: True if the dictionary is valid, False otherwise
    :rtype: bool
    """
    if 'text' not in question:
        _logger.debug("`text` key not found in dictionary")
        return False
    if 'image' not in question:
        _logger.debug("`image` key not found in dictionary")
        return False
    if not isinstance(question['text'], str):
        _logger.debug("Value for `text` is not a string")
        return False
    if not isinstance(question['image'], list):
        _logger.debug("Value for `image` is not a list")
        return False
    for img in question['image']:
        if not isinstance(img, str):
            _logger.debug("Value in `image` list is not a string")
            return False
        if not _image_verification(img):
            _logger.debug("Value in `image` list is not a valid image (url or base64)")
            return False
    
    text_segs = question['text'].split(_image_special_token)
    if not len(text_segs) == len(question['image']) + 1:
        _logger.debug("Number of images does not match the number of image placeholders in the text")
        return False
    return True


def _append_question(context: list, question: Union[str, Dict[str, Union[str, List[str]]]]):
    """
    Append a question to the context list

    :param context: List where the question will be appended
    :type context: list

    :param question: Question to append
    :type question: Union[str, Dict[str, Union[str, List[str]]]]

    :return: None
    """
    _logger.debug(f"Appending question to context dictionary")
    if isinstance(question, dict):
        if not _vision_question_verification(question):
            raise ValueError("Invalid question dictionary, expected a dictionary with 'text' and 'image' keys.")
        
        content_list = []
        text_segs = question['text'].split(_image_special_token)
        for i, seg in enumerate(text_segs):
            if seg:
                content_list.append({"type": "text", "text": seg})
            if i < len(question['image']):
                content_list.append({"type": "image_url", "image_url": {"url": question['image'][i]}})
        
        context.append({"role": "user", "content": content_list})
    else:
        context.append({"role": "user", "content": question})


def _append_response(context: list, response: str):
    """
    Append a response from GPT to the context list

    :param context: List where the response will be appended
    :type context: list

    :param response: Response to append
    :type response: str

    :return: None
    """
    _logger.debug(f"Appending response to context")
    context.append({"role": "assistant", "content": response})


def _extract_raw_result(raw: ChatCompletion) -> str:
    """
    Extract the response from the ChatCompletion object

    :param raw: ChatCompletion object from OpenAI
    :type raw: ChatCompletion

    :return: Extracted response
    :rtype: str
    """
    _logger.debug(f"Extracting response from ChatCompletion object")
    result = raw.choices[0].message.content.strip() if raw.choices[0].message.content else ""
    if len(result) == 0:
        _logger.warning(f"Empty response from API - finish_reason: {raw.choices[0].finish_reason if raw.choices else 'unknown'}")
    return result


def generate_explanation(questions: list,
                         model_name: str = _default_model,
                         verbose: bool = False,
                         task_desc: str = None,
                         debug_log: str = None,
                         client: OpenAI = None,
                         init_context: str = None,
                         **kwargs) -> Dict[str, str]:
    """
    Given a list of questions for a given context, generate responses for each question step by step.

    :param questions: List of questions to ask
    :type questions: List[Union[str, Dict[str, List[str]]]]

    :param model_name: Name for the OpenAI model to use
    :type model_name: str(, optional)

    :param verbose: Print chat history if set to True
    :type verbose: bool(, optional)

    :param task_desc: System prompt
    :type task_desc: str(, optional)

    :param debug_log: Path to save the debug log
    :type debug_log: str(, optional)

    :param client: OpenAI client object, defaults to global _client if None
    :type client: OpenAI(, optional)

    :param init_context: Part of the first question (deprecated)
    :type init_context: str(, optional)

    :param kwargs: Generation parameters
    :type kwargs: dict(, optional)

    :return: Dictionary with responses for each question
    :rtype: dict
    """
    # Use global client if none provided
    if client is None:
        client = _client
    
    # Check if the utility is properly setup
    assert client is not None, "need to run init() first or provide client."

    # Create list to hold chat histories
    context = list()

    # Create return dict object
    output_dict = dict()

    # Add system prompt if provided
    if task_desc is not None:
        _logger.debug(f"Adding system prompt: {task_desc}")
        context.append({"role": "system", "content": task_desc})

    # Add the initial context to the first question if provided
    if init_context:
        warn("arg: init_context will be deprecated in version 0.4, please put the initial context in the first"
             "item of the questions list.", DeprecationWarning)
        _logger.info(f"Prepend initial context to the first question")
        if isinstance(questions[0], dict):
            questions[0]['text'] = '\n\n'.join([init_context, questions[0]['text']]).strip()
        else:
            questions[0] = '\n\n'.join([init_context, questions[0]]).strip()
        output_dict['initial_context'] = init_context

    # Loop through the questions
    for q_id, q in enumerate(questions):
        _logger.debug(f"Appending question: {q}")
        # Add the question to the context
        _append_question(context, q)
        _logger.debug(f"Context: {context}")
        # Get the response from OpenAI
        _logger.info("Getting response from OpenAI")
        curr_response = _extract_raw_result(_get_response(context, model_name, debug_log, client, **kwargs))
        # Append the response to the context
        _logger.debug(f"Appending response: {curr_response}")
        _append_response(context, curr_response)

        # Record response to the question in output_dict
        _logger.debug("Saving response to output_dict")
        if isinstance(q, dict):
            if q_id == 0 and init_context is not None:
                output_dict[q['text'].replace(init_context, '').strip()] = curr_response
            else:
                output_dict[q['text']] = curr_response
        else:
            if q_id == 0 and init_context is not None:
                output_dict[q.replace(init_context, '').strip()] = curr_response
            else:
                output_dict[q] = curr_response

    # If verbose was set, print the entire chat history
    if verbose:
        pprint(context)
        print()
    return output_dict


def generate_explanation_wrapper(arg_dict: dict) -> Dict[str, str]:
    """
    Wrapper function of `generate_explanation` for threading.
    Creates thread-local client.

    :param arg_dict: Argument dictionary for `generate_explanation`
    :type arg_dict: dict

    :return: Dictionary with responses for each question
    :rtype: dict
    """
    # Get thread-local OpenAI client
    _logger.debug("Getting thread-local OpenAI client")
    openai_args = arg_dict.get('openai_args', {})
    api_key = openai_args.get('api_key')
    
    # Remove api_key from openai_args to avoid duplication
    other_args = {k: v for k, v in openai_args.items() if k != 'api_key'}
    
    client = _get_thread_client(api_key, **other_args)
    
    # Remove openai_args and add client
    if 'openai_args' in arg_dict:
        del arg_dict['openai_args']
    arg_dict['client'] = client
    
    try:
        _logger.debug("Calling generate_explanation in thread")
        return generate_explanation(**arg_dict)
    except Exception as e:
        _logger.error(f"Error in generate_explanation: {e}")
        _logger.error("Returning empty dictionary")
        return dict()


def init(api_key: str, gen_conf: dict = None, **kwargs):
    """
    Initialize the OpenAI utility with the given API key and generation configuration.
    
    :param api_key: OpenAI API Key
    :type api_key: str

    :param gen_conf: Generation configuration to override the default configuration
    :type gen_conf: dict(, optional)

    :param kwargs: Other arguments to pass to OpenAI client
    :type kwargs: dict

    :return: None
    """
    global _generation_config
    global _client
    global _async_client

    _logger.info("Initializing Thread-GPT-Suite")
    # Check version compatibility
    assert _version_checker(_target_majors), \
        f"openai({openai.__version__}) is no longer supported, please upgrade first."
    
    # Initialize OpenAI Objects
    if not kwargs:
        _logger.info("Initializing OpenAI clients with default arguments (max_retries=5)")
        _client = OpenAI(api_key=api_key, max_retries=5)
        _async_client = AsyncOpenAI(api_key=api_key, max_retries=5)
    else:
        _logger.info("Initializing OpenAI clients with custom arguments")
        _client = OpenAI(api_key=api_key, **kwargs)
        _async_client = AsyncOpenAI(api_key=api_key, **kwargs)
    
    # Override configuration files if needed
    if gen_conf:
        for k in list(gen_conf.keys()):
            if k in _generation_config:
                _logger.info(f"Overriding generation configuration: {k}={gen_conf[k]}")
                _generation_config[k] = gen_conf[k]
