import logging
import re
import inspect
import hydra
from litellm import completion
import time
from litellm import completion
import os
import logging
import concurrent.futures
import time
import re
import inspect

def init_client(cfg):
    global client
    if cfg.get("model", None): # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_client.openai import OpenAIClient
            client = OpenAIClient(model, temperature)
        elif cfg.model.startswith("GLM"):
            from utils.llm_client.zhipuai import ZhipuAIClient
            client = ZhipuAIClient(model, temperature)
        else: # fall back to Llama API
            from utils.llm_client.llama_api import LlamaAPIClient
            client = LlamaAPIClient(model, temperature)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client
    

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):

    assert isinstance(messages_list, list), "messages_list should be a list."
    try:
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
    except:
        print(messages_list)
        raise IndexError("Something is wrong.")

    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

    num_workers = os.cpu_count()
    if "gpt" not in model:
        # Transform messages if n > 1
        messages_list *= n
        n = 1
        num_workers = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(n, messages, temperature, model) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c.message.content)
    return contents


def chat_completion(n: int, messages: list[dict], temperature: float,
                    model: str = "nvidia_nim/openai/gpt-oss-120b", deadline=None) -> list[dict]:
    """Call the configured model with bounded retries and a shared deadline."""
    from source.search_budget import BudgetExhausted

    for attempt in range(3):
        timeout = 60.0
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                raise BudgetExhausted("Search wall-clock budget exhausted during generation.")
        try:
            response = completion(model=model, messages=messages,
                                  temperature=temperature, n=n,
                                  timeout=timeout, num_retries=0)
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Search wall-clock budget exhausted during generation.")
            return response.choices
        except BudgetExhausted:
            raise
        except Exception:
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Search wall-clock budget exhausted during generation.")
            if attempt == 2:
                raise
            delay = 3 if deadline is None else min(3, max(0, deadline - time.monotonic()))
            logging.warning("LLM request failed (attempt %s); retrying.", attempt + 1)
            time.sleep(delay)


def format_messages(cfg, pre_messages):
    messages = [{"role": "system", "content": pre_messages["system"]},
                {"role": "user", "content": pre_messages["user"]}]
    return messages