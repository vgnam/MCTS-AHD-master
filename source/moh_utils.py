"""Response-parsing helpers used by MoH and by the optimizer code it generates.

Ported from ``utils/utils.py`` of https://github.com/yiding-s/MoH. Generated
optimizers import them with ``from source.moh_utils import extract_code, extract_idea``.
"""

import re


def code_only(code_string):
    """Remove # comments and redundant blank lines."""
    code_without_comments = re.sub(r"#.*", "", code_string)
    cleaned_code = re.sub(r"\n\s*\n", "\n", code_without_comments)
    return cleaned_code.strip()


def clean_code(algorithm_str):
    if isinstance(algorithm_str, str):
        return code_only(algorithm_str)
    if isinstance(algorithm_str, list):
        return [code_only(s) if isinstance(s, str) else s for s in algorithm_str]
    return algorithm_str


def find_braces(response):
    """Content of the first {...} pair (the one-sentence idea)."""
    if not isinstance(response, str):
        return None
    match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if match:
        return match.group(1)
    if "import" in response:
        return response.split("import")[0]
    return None


def extract_idea(algorithm_str):
    if isinstance(algorithm_str, list):
        return [extract_idea(s) for s in algorithm_str]
    return find_braces(algorithm_str)


def find_largest_code_block_line_by_line(text):
    """Largest ``` fenced block in ``text`` (without the fences), or None."""
    if not isinstance(text, str):
        return None
    largest_block, current_block, nesting_level = "", "", 0
    for line in text.split("\n"):
        if line.startswith("```"):
            if not line[3:].strip():  # closing delimiter
                nesting_level -= 1
                current_block += line + "\n"
                if nesting_level == 0:
                    if len(current_block) > len(largest_block):
                        largest_block = current_block
                    current_block = ""
            else:  # opening delimiter
                current_block += line + "\n"
                nesting_level += 1
        elif nesting_level > 0:
            current_block += line + "\n"
    if largest_block:
        largest_block = "\n".join(largest_block.strip().split("\n")[1:-1])
    return largest_block or None


def extract_code(algorithm_str):
    if isinstance(algorithm_str, list):
        return [extract_code(s) for s in algorithm_str]
    return find_largest_code_block_line_by_line(algorithm_str)


def find_txt_block(string):
    """Content of a ```txt fenced block."""
    inside, block = False, ""
    for line in (string or "").split("\n"):
        if line.strip() == "```txt":
            inside, block = True, line + "\n"
        elif line.strip() == "```" and inside:
            block += line + "\n"
            inside = False
        elif inside:
            block += line + "\n"
    return block


def match_number(string):
    """First integer in ``string`` (1 if none)."""
    match = re.search(r"\d+", string or "")
    return int(match.group()) if match else 1
