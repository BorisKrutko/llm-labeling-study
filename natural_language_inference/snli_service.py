import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from IPython.display import display


def get_model_summary(df: pd.DataFrame, model_name: str, gold_label='gold_label', pred_label='pred_label') -> pd.Series:
    total_support = len(df)
    nan_count = df[pred_label].isna().sum()
    cleaned_df = df.dropna(subset=[pred_label]).copy()

    y_true = cleaned_df[gold_label]
    y_pred = cleaned_df[pred_label]
    
    labels = sorted(y_true.unique())

    report_dict = classification_report(y_true, y_pred, output_dict=True, labels=labels)

    accuracy = report_dict.get('accuracy', accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    correct_predictions_by_class = cm.diagonal()
    
    summary = {
        'Model': model_name,
        'NaN_Count': nan_count,
        'Total_Support': total_support,
        
        'Accuracy': accuracy,
        'Support_Cleaned': report_dict['weighted avg']['support'], 
        
        'Total Correct Predictions': int(round(report_dict['weighted avg']['support'] * accuracy)),
        
        'Macro Precision': report_dict['macro avg']['precision'],
        'Macro Recall': report_dict['macro avg']['recall'],
        'Macro F1': report_dict['macro avg']['f1-score'],
        
        'Weighted Precision': report_dict['weighted avg']['precision'],
        'Weighted Recall': report_dict['weighted avg']['recall'],
        'Weighted F1': report_dict['weighted avg']['f1-score'],
    }
    
    for i, label in enumerate(labels):
        summary[f'Correct_{label}'] = correct_predictions_by_class[i]
        
    return pd.Series(summary)

def calculate_final_score(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    
    max_cost = df['Total_Cost'].max()
    if max_cost == 0: max_cost = 1.0
    df['Cost_Penalty'] = (df['Total_Cost'] / max_cost) * 0.2
    df['NaN_Penalty'] = df['NaN_Count'] / df['Total_Support']
    
    df['Final_Score'] = (
        df['Weighted F1'] 
        - df['Cost_Penalty'] 
        - df['NaN_Penalty']
    )
    
    df = df.sort_values(by='Final_Score', ascending=False)
    

    cols = ['Model', 'Weighted F1', 'Total_Cost', 'NaN_Count', 'Cost_Penalty', 'NaN_Penalty', 'Final_Score']
    display(df[cols].round(4))
    
    best_model = df.iloc[0]['Model']
    print(f"\nBest model: {best_model}")
    
    return df


