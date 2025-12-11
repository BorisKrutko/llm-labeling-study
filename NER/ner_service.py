import pandas as pd
import ast
from collections import Counter


def safe_literal_eval(val):
    # "['a']" -> ['a'].
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return []


def normalize_entity(e):
    """
    entities -> ('text', 'CLASS') 
    """
    if not isinstance(e, dict): 
        return None
    
    txt = str(e.get("entity", "")).strip().lower()
    cls = str(e.get("class", "")).strip().upper()
    
    if not txt:
        return None
        
    return (txt, cls)

def calc_row_metrics(true_list, pred_list):
    """
    Precision/Recall/F1 for 1 row
    [] -> (F1=1.0).
    """
    # 1. normalize_entity
    true_norm = [normalize_entity(e) for e in true_list if normalize_entity(e)]
    pred_norm = [normalize_entity(e) for e in pred_list if normalize_entity(e)]

    if not true_norm and not pred_norm:
        return 1.0, 1.0, 1.0, [], []

    # 2. Counting matches
    true_counts = Counter(true_norm)
    pred_counts = Counter(pred_norm)

    tp = sum((true_counts & pred_counts).values()) # Пересечение
    fp = sum((pred_counts - true_counts).values()) # Лишнее
    fn = sum((true_counts - pred_counts).values()) # Не хватило

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    missing = list((true_counts - pred_counts).elements()) 
    extra = list((pred_counts - true_counts).elements())   

    return precision, recall, f1, missing, extra


def evaluate_model(pred_df, golden_df, model_name, price_dict):
    df_p = pred_df.copy()
    df_g = golden_df.copy()

    # 1. "" -> []
    df_p['entities_pred'] = df_p['entities_pred'].apply(safe_literal_eval)
    
    if isinstance(df_g['entities'].iloc[0], str):
        df_g['entities'] = df_g['entities'].apply(safe_literal_eval)

    # 2. merge golden row <-> model row
    merged = pd.merge(df_p, df_g[['id', 'entities']], on='id', how='inner')
    
    if merged.empty:
        print(f"{model_name}: No matches for id")
        return None

    # 3 calc metrics for row
    metrics = merged.apply(
        lambda row: calc_row_metrics(row['entities'], row['entities_pred']),
        axis=1,
        result_type='expand'
    )
    metrics.columns = ["precision", "recall", "f1", "missing", "extra"]
    
    full_df = pd.concat([merged, metrics], axis=1)

    # 4. calc res metrics (mean)
    avg_f1 = full_df['f1'].mean()
    avg_prec = full_df['precision'].mean()
    avg_rec = full_df['recall'].mean()
    avg_time = full_df['llm_time'].mean() if 'llm_time' in full_df else 0

    # 5. price
    prices = price_dict.get(model_name, {'prompt': 0, 'completion': 0})
    p_tok = full_df['llm_prompt_tokens'].mean() if 'llm_prompt_tokens' in full_df else 0
    c_tok = full_df['llm_completion_tokens'].mean() if 'llm_completion_tokens' in full_df else 0
    
    avg_cost = (p_tok * prices['prompt']) + (c_tok * prices['completion'])

    return {
        "model": model_name,
        "f1": avg_f1,
        "precision": avg_prec,
        "recall": avg_rec,
        "avg_time_sec": avg_time,
        "avg_cost_usd": avg_cost,
        "count": len(full_df)
    }

def run_comparison(dfs, golden_df, price_dict):
    results = []
    golden_prepared = golden_df.copy()
    golden_prepared['entities'] = golden_prepared['entities'].apply(safe_literal_eval)
        
    for name, df in dfs.items():
        stats = evaluate_model(df, golden_prepared, name, price_dict)
        
        if stats:
            score = stats['f1']
            
            stats['final_score'] = score
            results.append(stats)
            

    res_df = pd.DataFrame(results)    
    res_df = res_df.sort_values(by=['f1', 'avg_cost_usd'], ascending=[False, True])
    
    return res_df
