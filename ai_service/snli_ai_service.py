import pandas as pd
from tqdm import tqdm
import copy


def parse_sentiment(text):
    text = text.lower()
    
    if "entailment" in text and not ("contradiction" in text or "neutral" in text):
        return 'entailment'
    if "contradiction" in text and not ("entailment" in text or "neutral" in text):
        return 'contradiction'
    if "neutral" in text and not ("contradiction" in text or "entailment" in text):
        return 'neutral'
    
    return None

def snli_label_dataset(
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
    pt_list = [] 
    ct_list = [] 
    tt_list = [] 
        
    for index, row in tqdm(df_out.iterrows(), total=len(df_out)): 
        premise = row['sentence1']      # 'premise'
        hypothesis = row['sentence2']   # 'hypothesis' 

        try:
            formatted_content = prompt_template.format(
                premise=premise,
                hypothesis=hypothesis
            )
        except KeyError as e:
            print(f"Ошибка форматирования промпта: {e}")
            formatted_content = prompt_template 

        messages = [{"role": "user", "content": formatted_content}]
        raw_ans, elapsed, pt, ct = llm_classifier(messages, model_name)
        
        tt = pt + ct
        clean_ans = parse_sentiment(raw_ans)

        preds_raw.append(raw_ans)
        preds_label.append(clean_ans)
        times.append(elapsed)
        pt_list.append(pt)
        ct_list.append(ct)
        tt_list.append(tt)
       
    df_out['pred_label'] = preds_label    
    df_out['time_sec'] = times
    df_out['prompt_tokens'] = pt_list
    df_out['completion_tokens'] = ct_list
    df_out['total_tokens'] = tt_list

    filename = f"predictions_train_{model_name}.csv"
    df_out.to_csv(f'../natural_language_inference/{filename}', index=False)

    metrics = {
        'count': len(df_out),
        'avg_time': sum(times) / len(times) if times else 0,
        'total_cost_tokens': sum(tt_list),
        'sum_prompt_tokens': sum(pt_list),
        'sum_completion_tokens': sum(ct_list),
        'errors': preds_label.count(None)
    }
        
    return df_out, metrics