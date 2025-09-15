# Automated Prompt

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# Paths
model_name = "" #Deleted, but should be added
csv_path = ""   #Deleted, but should be added
output_csv_path = "" #Deleted, but should be added

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Read CSV
df = pd.read_csv(csv_path)

def get_llama_explanation(report, prediction):
    if prediction == 0:
        explanation_request = (
            "identify and explain the terminology or findings that would support NOT referring this patient to the Orthopedic Oncology department."
        )
    elif prediction == 1:
        explanation_request = (
            "identify and explain the key terminology or findings that would indicate the patient SHOULD be referred to the Orthopedic Oncology department."
        )
    elif prediction == 2:
        explanation_request = (
            "identify and explain the specific terminology or findings that suggest the patient should be referred to the Orthopedic Oncology department due to risk of pathological fracture (emergency)."
        )
    else:
        explanation_request = "identify and explain the relevant terminology."

    prompt = f"""
Given the radiology report below, {explanation_request}

Report:
<{report}>

LLAMA Explanation:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Optionally, clean up output
    return explanation.split("LLAMA Explanation:")[-1].strip()

# Apply to each row using predicted label
explanations = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    report = row['text']
    prediction = row['prediction']
    try:
        explanation = get_llama_explanation(report, prediction)
    except Exception as e:
        explanation = f"ERROR: {e}"
    explanations.append(explanation)

df['llama_explanation'] = explanations
df.to_csv(output_csv_path, index=False)
print(f"Done! Explanations saved to {output_csv_path}")
