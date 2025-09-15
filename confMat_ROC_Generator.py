# Confusion Matrix generation for 2x2 matrix (No Referral vs Referral)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os # Import the 'os' module to handle file paths and directories
import textwrap # Import the textwrap module for label wrapping

 
# --------------------------------------------------------------------------
# 1. DATA INPUT: Define your confusion matrix and class labels here
# --------------------------------------------------------------------------
# The confusion matrix values provided in your results
confusion_matrix_data = np.array([
    [37, 0],
    [0, 47]
])
 
# The labels for your classes (in the order they appear in the matrix)
class_labels = ["No Referral", "Referral"]
 
# --------------------------------------------------------------------------
# 2. VISUAL CUSTOMIZATION: Adjust all visual parameters here
# --------------------------------------------------------------------------
 
# --- Label Customization ---
# SET a maximum width (in characters) for each line of a label.
# The code will automatically wrap labels longer than this.
label_wrap_width = 13 # Adjust this number to control wrapping
 
# --- Font Customization ---
title_font = {'family': 'serif', 'size': 22, 'weight': 'bold'}
axis_label_font = {'family': 'sans-serif', 'size': 20, 'weight': 'normal'}
tick_label_font_size = 20
matrix_number_font_size = 20
 
# --- Color Customization ---
color_map = "Blues"
 
# --- Figure and File Settings ---
figure_size = (9, 7)
output_folder = "" #Deleted, but should be added
output_filename = "confusion_matrix_2class_wrapped.png"
output_dpi = 600
 
# --------------------------------------------------------------------------
# 3. PLOTTING LOGIC: This section generates and saves the plot
# --------------------------------------------------------------------------
 
# --- Handle Label Wrapping ---
# This list comprehension will process each label.
# textwrap.fill splits the string at the specified width and joins with a newline.
wrapped_labels = [textwrap.fill(label, width=label_wrap_width) for label in class_labels]
 
# --- Create Directory ---
os.makedirs(output_folder, exist_ok=True)
print(f"Output directory '{output_folder}' is ready.")
 
# --- Create Full File Path ---
full_save_path = os.path.join(output_folder, output_filename)
 
# --- Generate the Plot ---
plt.figure(figsize=figure_size)
 
heatmap = sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt='d',
    cmap=color_map,
    linewidths=.5,
    annot_kws={'size': matrix_number_font_size},
    # Pass the wrapped labels to the heatmap directly
    xticklabels=wrapped_labels,
    yticklabels=wrapped_labels
)
 
# --- Set Title and Axis Labels ---
# heatmap.set_title("Confusion Matrix (3-Class)", fontdict=title_font, pad=20)
heatmap.set_xlabel("Predicted Label", fontdict=axis_label_font, labelpad=15)
heatmap.set_ylabel("True Label", fontdict=axis_label_font, labelpad=15)
 
# --- Control Label Orientation ---
# Set rotation for x-axis labels to 0 for horizontal display.
# 'ha' (horizontal alignment) is set to 'center' for proper alignment.
plt.xticks(rotation=0, ha='center', fontsize=tick_label_font_size)
 
# Set rotation for y-axis labels to 90 for vertical display.
# 'va' (vertical alignment) is set to 'center' to ensure the label is centered on the tick.
plt.yticks(rotation=90, va='center', fontsize=tick_label_font_size)
 
 
# Ensure the layout is tight to prevent labels from being cut off
plt.tight_layout()
 
# --- Save and Show the Plot ---
try:
    plt.savefig(full_save_path, dpi=output_dpi, bbox_inches='tight')
    print(f"Confusion matrix saved successfully at: '{full_save_path}'")
except Exception as e:
    print(f"Error saving file: {e}")
 
# Display the plot
plt.show()




# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import textwrap # Import the textwrap module for label wrapping
 
# --------------------------------------------------------------------------
# 1. DATA INPUT: Define your 3x3 confusion matrix and class labels here
# --------------------------------------------------------------------------
# The 3x3 confusion matrix values from your results
confusion_matrix_data = np.array([
    [37, 0, 0],
    [0, 29, 0],
    [0, 8, 10]
])
 
# The labels for your three classes.
# The wrapping logic will handle long labels like "Referral/High Risk".
class_labels = ["No Referral", "Referral", "Referral/High Risk"]
 
# --------------------------------------------------------------------------
# 2. VISUAL CUSTOMIZATION: Adjust all visual parameters here
# --------------------------------------------------------------------------
 
# --- Label Customization ---
# SET a maximum width (in characters) for each line of a label.
# The code will automatically wrap labels longer than this.
label_wrap_width = 13 # Adjust this number to control wrapping
 
# --- Font Customization ---
title_font = {'family': 'serif', 'size': 22, 'weight': 'bold'}
axis_label_font = {'family': 'sans-serif', 'size': 20, 'weight': 'normal'}
tick_label_font_size = 20
matrix_number_font_size = 20
 
# --- Color Customization ---
color_map = "Blues"
 
# --- Figure and File Settings ---
figure_size = (9, 7)
output_folder = ""   #Deleted, but should be added
output_filename = "confusion_matrix_3class_wrapped.png"
output_dpi = 600
 
# --------------------------------------------------------------------------
# 3. PLOTTING LOGIC: This section generates and saves the plot
# --------------------------------------------------------------------------
 
# --- Handle Label Wrapping ---
# This list comprehension will process each label.
# textwrap.fill splits the string at the specified width and joins with a newline.
wrapped_labels = [textwrap.fill(label, width=label_wrap_width) for label in class_labels]
 
# --- Create Directory ---
os.makedirs(output_folder, exist_ok=True)
print(f"Output directory '{output_folder}' is ready.")
 
# --- Create Full File Path ---
full_save_path = os.path.join(output_folder, output_filename)
 
# --- Generate the Plot ---
plt.figure(figsize=figure_size)
 
heatmap = sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt='d',
    cmap=color_map,
    linewidths=.8,
    annot_kws={'size': matrix_number_font_size},
    # Pass the wrapped labels to the heatmap directly
    xticklabels=wrapped_labels,
    yticklabels=wrapped_labels
)
 
# --- Set Title and Axis Labels ---
# heatmap.set_title("Confusion Matrix (3-Class)", fontdict=title_font, pad=20)
heatmap.set_xlabel("Predicted Label", fontdict=axis_label_font, labelpad=15)
heatmap.set_ylabel("True Label", fontdict=axis_label_font, labelpad=15)
 
# --- Control Label Orientation ---
# Set rotation for x-axis labels to 0 for horizontal display.
# 'ha' (horizontal alignment) is set to 'center' for proper alignment.
plt.xticks(rotation=0, ha='center', fontsize=tick_label_font_size)
 
# Set rotation for y-axis labels to 90 for vertical display.
# 'va' (vertical alignment) is set to 'center' to ensure the label is centered on the tick.
plt.yticks(rotation=90, va='center', fontsize=tick_label_font_size)
 
 
# Ensure the layout is tight to prevent labels from being cut off
plt.tight_layout()
 
# --- Save and Show the Plot ---
try:
    plt.savefig(full_save_path, dpi=output_dpi, bbox_inches='tight')
    print(f"Confusion matrix saved successfully at: '{full_save_path}'")
except Exception as e:
    print(f"Error saving file: {e}")
 
# Display the plot
plt.show()









# ROC Curve generation

from datasets import Dataset as HFDataset 
from sklearn.metrics import ( 
    roc_curve, 
    auc, 
    roc_auc_score # Added for AUC calculation in metrics 
) 


# --- Load Precomputed ROC Curve Data ---
import numpy as np
import matplotlib.pyplot as plt
 
# Specify the directory where the .npy files are saved
roc_data_folder = "" #Deleted, but should be added
output_folder = ""   #Deleted, but should be added
# Load ROC curves
fpr = np.load(os.path.join(roc_data_folder, "fpr0_12.npy"))
tpr = np.load(os.path.join(roc_data_folder, "tpr0_12.npy"))
fpr12 = np.load(os.path.join(roc_data_folder, "fpr1_2.npy"))
tpr12 = np.load(os.path.join(roc_data_folder, "tpr1_2.npy"))
 
# Recalculate AUCs
roc_auc = auc(fpr, tpr)
roc_auc12 = auc(fpr12, tpr12)
 
print(f"\nLoaded Ensemble ROC AUC (No Referral vs. Referral/High Risk): {roc_auc:.4f}")
print(f"Loaded Ensemble ROC AUC (Referral vs. High Risk): {roc_auc12:.4f}")
 
# --- Plot Combined ROC Curves ---
plt.figure(figsize=(9, 7))
plt.plot(fpr, tpr, color="blue", lw=4, label=f"No Referral vs Referral/High Risk (AUC = {roc_auc:.2f})")
plt.plot(fpr12, tpr12, color="red", lw=4, label=f"Referral vs Referral/High Risk (AUC = {roc_auc12:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
# plt.title("Final ROC Curves from Ensemble Model (on Hold-Out Test Set)", fontsize=14)
plt.legend(loc="lower right", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

save_path = os.path.join(output_folder, "combined_roc_curves.png")
plt.savefig(save_path, dpi=output_dpi)
plt.close()
 
print(f"\nCombined ROC curves figure saved to: {save_path}") 