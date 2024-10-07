import json
import re


def parse_dict_keys(dict_raw: dict, substring: str = '_') -> dict:
    """
    Split by substring and capitalize dictionary keys
    :param dict_raw: dict
    :param substring: str
    :return: dict
            Dictionary with capitalized keys
    """
    dict_parsed = {}
    for key, value in dict_raw.items():
        new_key = key.upper().replace(substring, ' ')
        if isinstance(value, list):
            if isinstance(value[0], dict):
                # Parse embedded dictionaries
                value = [parse_dict_keys(item_dict) for item_dict in value]
        dict_parsed[new_key] = value
    return dict_parsed


def parse_to_text(str_response):
    """
    Converts a string representation of a JSON object, potentially enclosed in triple ticks, into a text object.
    Args:str_response (str): A string that contains a JSON object, potentially enclosed within triple ticks.
    Returns:str or None: The parsed JSON object as a dictionary if parsing is successful; None otherwise.
    """
    # Remove triple ticks
    clean_str = str_response.replace("###", "").strip()
    clean_str = clean_str.replace("```", "").strip()

    return clean_str


def parse_to_json(str_response):
    """
    Converts a string representation of a JSON object, potentially enclosed in triple ticks, into a JSON object.

    Args:
        str_response (str): A string that contains a JSON object, potentially enclosed within triple ticks.

    Returns:
        dict or None: The parsed JSON object as a dictionary if parsing is successful; None otherwise.

    Raises:
        Prints an error message if JSON decoding fails.
    """
    # Remove triple ticks
    clean_str = str_response.replace("```json", "").strip()
    clean_str = clean_str.replace("```", "").strip()

    # Replace single quotes with double quotes if necessary
    # This step might need adjustments based on the actual format of your response
    # clean_str = clean_str.replace("'", '"')

    try:
        # Parse the string into JSON
        json_data = json.loads(clean_str)
        return json_data
    except json.JSONDecodeError as e:
        # Handle parsing error (invalid JSON)
        print(f"Error parsing JSON: {e}")
        return None


def parse_markdown(md_content):
    """
    Extracts the first Markdown table found in the given string.

    Parameters:
    - md_content (str): A string that may contain a Markdown table along with other content.

    Returns:
    - str: The extracted Markdown table, or an empty string if no table is found.
    """
    # Regex pattern to match a Markdown table
    # Explanation:
    # - Look for lines starting with '|' or digits/spaces (for the header and rows)
    # - The table ends when a line does not start with '|', digit, or space
    # - Use re.DOTALL to match across multiple lines, including newlines
    table_pattern = r'((?:^\|.*|\d+\s*\|.*)(?:\n|$))+'

    # Search for the pattern in the input string
    match = re.search(table_pattern, md_content, re.MULTILINE)

    # If a table is found, return it; otherwise, return an empty string
    return match.group(0) if match else ''
