import json

def extract_language_names(lang_str):
    try:
        lang_list = json.loads(lang_str)
        return [lang["name"] for lang in lang_list if "name" in lang]
    except (json.JSONDecodeError, TypeError):
        return []
