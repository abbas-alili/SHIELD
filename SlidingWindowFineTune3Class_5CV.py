# With Editable ROC Curves and 5-Fold Cross-Validation 

##### Sliding Window Data Preparation (3-Class Version) ##### 
import pandas as pd 
from collections import Counter 
from sklearn.model_selection import train_test_split, StratifiedKFold 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import os 
import torch 
from torch.nn.functional import softmax 
from transformers import ( 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EvalPrediction 
) 

from datasets import Dataset as HFDataset 
from sklearn.metrics import ( 
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    roc_auc_score # Added for AUC calculation in metrics 
) 

  

# ============================================================================== 
# 1. Data Loading & Global Configuration 
# ============================================================================== 
# --- CONFIGURATION --- 
BASE_OUTPUT_DIR = "" #Deleted, but should be added
MODEL_PATH = ""      #Deleted, but should be added
CSV_PATH = ""        #Deleted, but should be added
MAX_LEN = 512 
STRIDE = 256 
N_SPLITS = 5  # Number of folds for cross-validation 

# Create the main output directory 

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True) 

  

# --- DATA LOADING --- 
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1") 
df = df.iloc[1:557]  # Adjust this as needed for your data 
labels = df.iloc[:, 0].astype(int).tolist() 
texts = df.iloc[:, 1].astype(str).tolist() 

  

print(f"Loaded {len(texts)} total samples.") 
print(f"Initial Class Distribution: {Counter(labels)}") 

  

  

# ============================================================================== 
# 2. Hold-Out Test Set and Cross-Validation Preparation 
# ============================================================================== 

X_train_val, X_test, y_train_val, y_test = train_test_split( 
    texts, labels, 
    test_size=0.15, 
    stratify=labels, 
    random_state=42 
) 

  

print("\n--- Data Splitting ---") 
print(f"Training/Validation set size: {len(X_train_val)}") 
print(f"Hold-out Test set size: {len(X_test)}") 
print(f"Hold-out Test set balance: {Counter(y_test)}") 

  
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42) 

  
# ============================================================================== 
# 3. Helper Functions 
# ============================================================================== 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 

def sliding_tokenize(texts, labels): 
    """Tokenizes text data with a sliding window approach.""" 
    # ... (rest of the function is unchanged) 
    all_input_ids, all_attention_masks, all_labels = [], [], [] 
    for text, label in zip(texts, labels): 
        encodings = tokenizer( 
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LEN, 
            stride=STRIDE, 
            return_overflowing_tokens=True, 
            return_attention_mask=True 
        ) 
        all_input_ids.extend(encodings["input_ids"]) 
        all_attention_masks.extend(encodings["attention_mask"]) 
        all_labels.extend([label] * len(encodings["input_ids"])) 

    return HFDataset.from_dict({ 
        "input_ids": all_input_ids, 
        "attention_mask": all_attention_masks, 
        "labels": all_labels 
    }) 

  
def compute_metrics(p: EvalPrediction): 
    """ 
    Computes evaluation metrics, now including Macro AUC, for the Trainer. 
    """ 
    preds_logits = p.predictions 
    preds_probs = softmax(torch.tensor(preds_logits), dim=-1).numpy() 
    preds_labels = preds_probs.argmax(axis=1) 
    true_labels = p.label_ids 
  
    # Standard metrics 
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds_labels, average="macro") 
    acc = accuracy_score(true_labels, preds_labels) 

    # AUC Score 

    try: 
        # Handles case where only one class is present in a batch 
        auc_score = roc_auc_score(true_labels, preds_probs, multi_class='ovr', average='macro') 

    except ValueError: 
        auc_score = 0.0 # Assign a default value if AUC can't be computed 

  

    return { 
        "accuracy": acc, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "auc": auc_score 
    } 

  

# ============================================================================== 
# 4. 5-Fold Cross-Validation Training Loop 
# ============================================================================== 
all_fold_probs = [] 
all_fold_val_metrics = [] 
# NEW: Store classification reports from each fold's evaluation on the hold-out test set 
all_fold_majority_reports = [] 
all_fold_mean_max_reports = [] 

  
test_dataset = sliding_tokenize(X_test, y_test) 
X_train_val = np.array(X_train_val) 
y_train_val = np.array(y_train_val) 

  

# Recreate window counts for the HOLD-OUT TEST SET (needed for per-fold aggregation) 
window_counts_test = [len(tokenizer(text, return_overflowing_tokens=True, stride=STRIDE, max_length=MAX_LEN, truncation=True)['input_ids']) for text in X_test] 

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)): 
    print(f"\n===== Starting Fold {fold+1}/{N_SPLITS} =====") 
    X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx] 
    y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx] 
    train_dataset_fold = sliding_tokenize(X_train_fold, y_train_fold) 
    val_dataset_fold = sliding_tokenize(X_val_fold, y_val_fold) 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3) 
    fold_output_dir = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold+1}") 

    training_args = TrainingArguments( 
        output_dir=fold_output_dir, 
        num_train_epochs=8, 
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        logging_steps=10, 
        learning_rate=2e-5, 
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        report_to="none", 
        save_total_limit=1 
    ) 

  
    trainer = Trainer( 
        model=model, 
        args=training_args, 
        train_dataset=train_dataset_fold, 
        eval_dataset=val_dataset_fold, 
        compute_metrics=compute_metrics, 
    ) 
    trainer.train() 

  
    # --- Store validation metrics for CV summary --- 
    val_metrics = trainer.evaluate(eval_dataset=val_dataset_fold) 
    cleaned_metrics = {key.replace('eval_', ''): value for key, value in val_metrics.items()} 
    all_fold_val_metrics.append(cleaned_metrics) 
    print(f"Fold {fold+1} Validation Metrics: {cleaned_metrics}") 


    # --- Evaluate each fold's model on the HOLD-OUT TEST SET --- 
    print(f"--- Evaluating Fold {fold+1} model on the Hold-Out Test Set ---") 
    preds_output = trainer.predict(test_dataset) 
    fold_probs = softmax(torch.tensor(preds_output.predictions), dim=1).numpy() 
    all_fold_probs.append(fold_probs) # For final ensemble 
  

    # --- NEW: Aggregate predictions and generate report FOR THIS FOLD --- 
    start = 0 
    majority_preds_fold = [] 
    mean_max_preds_fold = [] 

    for count in window_counts_test: 
        group_probs = fold_probs[start:start+count] 
        group_preds = np.argmax(group_probs, axis=1) 
        majority_preds_fold.append(np.bincount(group_preds).argmax()) 
        mean_max_preds_fold.append(np.argmax(group_probs.mean(axis=0))) 
        start += count 

  

    # Get classification reports as dictionaries and store them 
    report_mj = classification_report(y_test, majority_preds_fold, output_dict=True) 
    report_mm = classification_report(y_test, mean_max_preds_fold, output_dict=True) 
    all_fold_majority_reports.append(report_mj) 
    all_fold_mean_max_reports.append(report_mm) 
    best_model_path = os.path.join(fold_output_dir, "best_model") 
    trainer.save_model(best_model_path) 
    tokenizer.save_pretrained(best_model_path) 
    print(f"Best model for Fold {fold+1} saved to {best_model_path}") 


# ============================================================================== 
# 5. Cross-Validated Model Performance Summary 
# ============================================================================== 

CV_SUMMARY_DIR = os.path.join(BASE_OUTPUT_DIR, "cross_validation_summary") 
os.makedirs(CV_SUMMARY_DIR, exist_ok=True) 
cv_metrics_df = pd.DataFrame(all_fold_val_metrics) 
summary_df = pd.DataFrame({'Average': cv_metrics_df.mean(), 'Std Dev': cv_metrics_df.std()}) 
print("\n\n===== Cross-Validation Performance Summary (on Validation Sets) =====") 
print(summary_df) 

  

# Save the summary report 

# ... (this part is largely unchanged but will now include AUC) 
with open(os.path.join(CV_SUMMARY_DIR, "cv_summary_report.txt"), "w") as f: 
    f.write("Cross-Validation Performance Summary\n...\n") # Abridged for brevity 
    f.write(summary_df.to_string(float_format="%.4f")) 

  

# Visualizations (Bar Chart and Histogram for F1 and AUC) 
for metric_to_plot in ['f1', 'auc']: 
    plt.figure(figsize=(10, 6)) 
    scores = cv_metrics_df[metric_to_plot] 
    avg_score = summary_df.loc[metric_to_plot, 'Average'] 
    plt.bar(range(1, N_SPLITS + 1), scores, label='Fold Score') 
    plt.axhline(y=avg_score, color='r', linestyle='--', label=f'Average ({metric_to_plot.upper()}) = {avg_score:.4f}') 
    plt.title(f"Cross-Validation {metric_to_plot.upper()} Scores per Fold", fontsize=14) 
    plt.savefig(os.path.join(CV_SUMMARY_DIR, f"cv_{metric_to_plot}_scores_barchart.png")) 
    plt.close() 

  
# ============================================================================== 
# 6. Aggregate Predictions and Final Evaluation on Hold-Out Set 
# ============================================================================== 

FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "final_evaluation") 
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True) 

# --- NEW: Save the hold-out test dataset for reference --- 
test_df = pd.DataFrame({'text': X_test, 'true_label': y_test}) 
test_set_path = os.path.join(FINAL_OUTPUT_DIR, "hold_out_test_set.csv") 
test_df.to_csv(test_set_path, index=False) 
print(f"\nHold-out test set saved to: {test_set_path}") 

  
# --- 6.1. Ensemble Predictions (Averaging probabilities across folds) --- 
mean_probs_over_folds = np.mean(all_fold_probs, axis=0) 

start = 0 
majority_preds_ensemble = [] 
mean_max_preds_ensemble = [] 
aggregated_report_probs = [] 

 

for count in window_counts_test: 
    group_probs = mean_probs_over_folds[start:start+count] 
    group_preds = np.argmax(group_probs, axis=1) 
    majority_preds_ensemble.append(np.bincount(group_preds).argmax()) 
    mean_probs = group_probs.mean(axis=0) 
    mean_max_preds_ensemble.append(np.argmax(mean_probs)) 
    aggregated_report_probs.append(mean_probs) 
    start += count 


aggregated_report_probs = np.array(aggregated_report_probs) 
 

# --- NEW: Save the detailed predictions from the ENSEMBLE model to a CSV file --- 
predictions_df = pd.DataFrame({ 
    'report_text': X_test, 
    'true_label': y_test, 
    'majority_vote_pred': majority_preds_ensemble, 
    'mean_max_prob_pred': mean_max_preds_ensemble, 
    'prob_class_0': aggregated_report_probs[:, 0], 
    'prob_class_1': aggregated_report_probs[:, 1], 
    'prob_class_2': aggregated_report_probs[:, 2] 

}) 

predictions_csv_path = os.path.join(FINAL_OUTPUT_DIR, "aggregated_predictions.csv") 
predictions_df.to_csv(predictions_csv_path, index=False) 
print(f"Aggregated predictions saved to: {predictions_csv_path}") 

# --- 6.2. NEW: Generate and Save Mean +- Std Dev Performance Reports --- 
def process_and_save_mean_std_reports(list_of_reports, method_name, output_dir): 
    """Parses a list of classification report dicts, calculates mean/std, and saves.""" 
    # Flatten the list of dictionaries into a more manageable structure 
    flat_reports = [] 
    for report in list_of_reports: 
        flat_report = {} 
        for key, value in report.items(): 
            if isinstance(value, dict): 
                for sub_key, sub_value in value.items(): 
                    flat_report[f"{key}_{sub_key}"] = sub_value 
            else: 
                flat_report[key] = value 
        flat_reports.append(flat_report) 

     

    df = pd.DataFrame(flat_reports) 
     

    # Calculate mean and std 
    mean_df = df.mean() 
    std_df = df.std() 

    # Format for display: "mean ± std" 
    results = {} 
    for metric in mean_df.index: 
        results[metric] = f"{mean_df[metric]:.4f} ± {std_df[metric]:.4f}" 

  

    # Reconstruct a report-like structure 
    final_report_str = f"===== Averaged Performance Report: {method_name.upper()} =====\n" 
    final_report_str += f"(Metrics are Mean ± Std. Dev. over {N_SPLITS} folds on hold-out set)\n\n" 

    # Define the order and names for the report 
    class_names = ["No Referral", "Referral", "Referral/High Risk"] 
    metrics = ['precision', 'recall', 'f1-score', 'support'] 

    header = f"{'':<25}" + "".join([f"{name:<20}" for name in class_names]) + f"{'macro avg':<20}{'weighted avg':<20}\n" 
    final_report_str += header 

    for metric in metrics: 
        row = f"{metric:<25}" 
        for i, name in enumerate(class_names): 
            key = f"{i}_{metric}" 
            row += f"{results.get(key, 'N/A'):<20}" 
        macro_key = f"macro avg_{metric}" 
        weighted_key = f"weighted avg_{metric}" 
        row += f"{results.get(macro_key, 'N/A'):<20}{results.get(weighted_key, 'N/A'):<20}\n" 
        final_report_str += row 

         

    final_report_str += f"\nAccuracy: {results.get('accuracy', 'N/A')}\n" 
    print("\n" + final_report_str) 

     

    with open(os.path.join(output_dir, f"{method_name}_mean_std_report.txt"), "w") as f: 
        f.write(final_report_str) 

  

# Process and save reports for both aggregation methods 
process_and_save_mean_std_reports(all_fold_majority_reports, "majority_vote", FINAL_OUTPUT_DIR) 
process_and_save_mean_std_reports(all_fold_mean_max_reports, "mean_max_prob", FINAL_OUTPUT_DIR) 

  

# --- 6.3. FIX: Define and Generate Confusion Matrix for the ENSEMBLE model ---
 
# FIX: The helper function to plot the confusion matrix is re-introduced here.
def plot_confusion_matrix(y_true, y_pred, name):
    """Calculates, plots, and saves a confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Referral", "Referral", "Referral/High Risk"],
                yticklabels=["No Referral", "Referral", "Referral/High Risk"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Final Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()
 
# This call is now correct because it uses the helper function defined above.
# It provides a useful single snapshot of the ensembled performance.
plot_confusion_matrix(y_test, mean_max_preds_ensemble, "ensemble_mean_max_prob")
print(f"\nEnsemble confusion matrix saved to: {FINAL_OUTPUT_DIR}")
 
 
# ==============================================================================
# 7. Final ROC Curve Generation (Using Ensemble Probabilities)
# ==============================================================================
# This section correctly uses the robust ensembled probabilities
 
# --- 7.1. ROC: No Referral (0) vs. Referral/High Risk (1+2) ---
binary_true_0v12 = np.array([0 if label == 0 else 1 for label in y_test])
binary_probs_0v12 = aggregated_report_probs[:, 1] + aggregated_report_probs[:, 2]
fpr, tpr, _ = roc_curve(binary_true_0v12, binary_probs_0v12)
roc_auc = auc(fpr, tpr)
print(f"\nEnsemble Binary ROC AUC (No Referral vs. Referral/High Risk): {roc_auc:.4f}")
 
# --- 7.2. ROC: Referral (1) vs. Referral/High Risk (2) ---
relevant_idx = [i for i, label in enumerate(y_test) if label in [1, 2]]
y_true_1v2 = np.array([1 if y_test[i] == 2 else 0 for i in relevant_idx])
probs_1v2_relevant = aggregated_report_probs[relevant_idx]
 
# To avoid division by zero if both probabilities are zero
denominator = probs_1v2_relevant[:, 1] + probs_1v2_relevant[:, 2]
probs_1v2_normalized = np.divide(probs_1v2_relevant[:, 2], denominator, out=np.zeros_like(denominator), where=denominator!=0)
 
fpr12, tpr12, _ = roc_curve(y_true_1v2, probs_1v2_normalized)
roc_auc12 = auc(fpr12, tpr12)
print(f"Ensemble Referral vs High Risk ROC AUC: {roc_auc12:.4f}")

print(fpr) 
print(tpr) 
print(fpr12) 
print(tpr12) 
np.save("fpr0_12.npy", fpr) 
np.save("tpr0_12.npy", tpr) 
np.save("fpr1_2.npy", fpr12) 
np.save("tpr1_2.npy", tpr12) 

# --- 7.3. Plot Combined ROC Curves ---
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"No Referral vs Referral/High Risk (AUC = {roc_auc:.2f})")
plt.plot(fpr12, tpr12, color="red", lw=2, label=f"Referral vs Referral/High Risk (AUC = {roc_auc12:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Final ROC Curves from Ensemble Model (on Hold-Out Test Set)", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
 
save_path = os.path.join(FINAL_OUTPUT_DIR, "combined_roc_curves.png")
plt.savefig(save_path)
plt.close()
 
print(f"\nCombined ROC curves figure saved to: {save_path}") 