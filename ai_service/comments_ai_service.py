import pandas as pd
from tqdm import tqdm
import copy


def parse_sentiment(text):
    text = text.lower()
    if "positive" in text and "negative" not in text:
        return 1
    if "negative" in text and "positive" not in text:
        return 0
    return -1  

def label_dataset_comments(
        df: pd.DataFrame, 
        prompt_template: str, 
        model_name: str, 
        llm_classifier,
    ):

    if df is None or df.empty:
        return None, {}

    df_out = df.copy()
        
    preds_raw = []
    preds_label = []
    times = []
    pt_list = [] # prompt tokens
    ct_list = [] # completion tokens
    tt_list = [] # total tokens
        
    for index, row in tqdm(df_out.iterrows(), total=len(df_out)):
        text = row['text']
            
        messages = copy.deepcopy(prompt_template)
            
        formatted = False
        for msg in messages:
            if "{text}" in msg["content"]:
                msg["content"] = msg["content"].format(text=text)
                formatted = True
            
        if not formatted:
            messages.append({"role": "user", "content": text})

        raw_ans, elapsed, pt, ct = llm_classifier(messages, model_name)
        tt = pt + ct

        clean_ans = parse_sentiment(raw_ans)

        preds_raw.append(raw_ans)
        preds_label.append(clean_ans)
        times.append(elapsed)
        pt_list.append(pt)
        ct_list.append(ct)
        tt_list.append(tt)

    df_out['pred_raw'] = preds_raw        
    df_out['pred_label'] = preds_label    
    df_out['time_sec'] = times
    df_out['prompt_tokens'] = pt_list
    df_out['completion_tokens'] = ct_list
    df_out['total_tokens'] = tt_list

    filename = f"predictions_train_{model_name}.csv"
    df_out.to_csv(f'../sentiment_analysis/{filename}', index=False)

    metrics = {
        'count': len(df_out),
        'avg_time': sum(times) / len(times) if times else 0,
        'total_cost_tokens': sum(tt_list),
        'sum_prompt_tokens': sum(pt_list),
        'sum_completion_tokens': sum(ct_list),
        'errors': preds_label.count("None")
    }
        
    return df_out, metrics