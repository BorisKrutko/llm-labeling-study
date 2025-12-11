import pandas as pd
from tqdm import tqdm
import copy
import re
import json

def parse_llm_response(response_text):
    try:
        match = re.search(r"\{.*\}", response_text, flags=re.S)
        if not match:
            return []

        data = json.loads(match.group(0))
        if "entities" in data:
            return data["entities"]
        return []
    except Exception:
        return []

def build_prompt(prompt_template, text):
    prompt = copy.deepcopy(prompt_template)
    
    prompt["content"] = prompt["content"].format(text=text)
    return prompt

def label_dataset_ner(
        df: pd.DataFrame, 
        prompt_template: str, 
        call_model_fn, 
        model_name: str, 
        text_col:str = "sentence_text"
    ):
    preds = []
    times = []
    prompt_toks = []
    completion_toks = []

    for text in tqdm(df[text_col]):
        prompt = build_prompt(prompt_template, text)

        raw_text, elapsed, ptoks, ctoks = call_model_fn(prompt, model_name=model_name)

        entities = parse_llm_response(raw_text)

        preds.append(entities)
        times.append(elapsed)
        prompt_toks.append(ptoks)
        completion_toks.append(ctoks)

    df["entities_pred"] = preds
    df["llm_time"] = times
    df["llm_prompt_tokens"] = prompt_toks
    df["llm_completion_tokens"] = completion_toks

    return df