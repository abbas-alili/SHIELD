# 30 pairs heatmap generated

"""
Analysis script to compare clinician vs. LLM-generated medical terminology
for orthopedic oncology referrals.
 
This script implements a four-phase analysis:
1.  Preprocessing: Normalizing and cleaning the term lists.
2.  Lexical Analysis: Using set theory to find exact matches and differences.
3.  Semantic Analysis: Using BioBERT embeddings and cosine similarity to find
    conceptually similar terms, with heatmaps for the MOST and LEAST similar pairs.
4.  Hierarchical Analysis: A demonstration of mapping terms to a medical
    ontology to understand their relationships.
 
All generated plots are automatically saved to files.
"""
 
import re
import pandas as pd
import torch
import numpy as np
import nltk
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import warnings
 
# --- PHASE 0: SETUP AND INITIAL DATA ---
# Populated with the comprehensive radiology and LLM terms.
doctors_list = [
"Humerus",
"Radius",
"Ulna",
"Femur",
"Tibia",
"Fibula",
"Lytic",
"Sclerotic",
"Blastic",
"Mixed lytic",
"sclerotic",
"Malignancy",
"malignant"
"Metastatic",
"metastases",
"Suspicious bony lesion",
"Suspicious for progression of osseous disease",
"Loss of cortical integrity",
"Endosteal scalloping",
"Periosteal reaction",
"FDG avidity",
"uptake in bone",
"Increased SUV max",
"T1 hypointense T2 hyperintense",
"Increased uptake in bone",
"Hypermetabolic lesion in bone",
"Hypermetabolic osseous disease",
"Focus of mineralization",
"Infiltrative lesion",
"Enhancing lesions",
"Cortical thinning",
"Cortical erosion",
"Cortical destruction",
"Cortical breakthrough",
"(Trans)cortical permeation",
"Osseous destruction",
"Lytic lesion",
"PSMA avid osseous lesion",
"Osteonecrosis",
"Signal change (marrow)",
"Lucency",
"aggressive appearing",
"expansile",
"hyper enhancing",
"hypointense",
"marrow replacing",
"cortical thickening",
"marrow infiltration",
"neoplastic",
"transcortical soft tissue extension",
"destructive",
"cortical narrowing",
"endosteal erosion",
"cortical discontinuity",
"Lesion",
"patient is at (high) risk of pathologic fracture",
"Increased risk of pathologic fracture",
"Impending pathologic fracture",
"Predisposed to pathologic fracture",
"Loss of cortical integrity",
"Lytic changes",
"Lytic lesion",
"Destructive lesion",
"Moth-eaten appearance",
"Mottled",
"Concerning for instability",
"Cortical thinning",
"Cortical erosion",
"Cortical destruction",
"Cortical breakthrough",
"Cortical transgression",
"Cortical dehiscence",
"Osseous destruction",
"Osteolysis,"
"Periosteal edema",
"reaction",
"periostitis",
"Endosteal scalloping",
"Aggressive",
"aggressive appearing",
"permeative",
"possible pathologic fracture",
"cortical permeation",
"cortical disruption",
"cortical loss",
"osteonecrosis",
"marrow edema/signal",
"cortical discontinuity"
]
 
llm_list = [
"Mass-like lesion",
"Concern for soft tissue neoplasm",
"Recommendation for percutaneous biopsy",
"Pelvic pain",
"Lesion within the left femoral neck",
"Skeletal metastasis",
"Metastatic cancer",
"Edema",
"Low T1 and high T2-weighted signal",
"Sclerotic lesion",
"Distal femur",
"History of neoplasm",
"Active metastatic disease is not excluded",
"Expansile humeral shaft mass",
"Well-defined cystic focus",
"Decreased T1 and increased T2 signal intensity",
"Endosteal scalloping",
"Aggressive-appearing lytic lesion",
"Permeation through the lateral cortex",
"Periosteal reaction",
"Abnormal soft tissue in the intramedullary space",
"Widespread osseous metastatic disease",
"History of breast cancer",
"Proximal humerus",
"Indeterminate lucent lesions",
"Increase in size of the left supraclavicular lymph node",
"Metastatic anorectal squamous cell cancer",
"Lytic lesion",
"Asymmetrical sclerotic density",
"History of intramedullary rod and screw stabilization",
"Callus formation",
"Pathological fracture",
"Ill-defined",
"Osteolytic lesion",
"Lesion of the left proximal humerus",
"Involves the entire humeral head and into the metaphyseal region",
"Cortical thinning",
"Suspicion of some possible subtle cracks within the cortical region",
"Further evaluation with MRI with and without contrast recommended",
"Greater trochanteric",
"Lytic lesion",
"Possible minimal labral calcification",
"Expansile destructive right proximal humerus mass",
"Pathologic fracture",
"Soft tissue component extends significantly beyond the cortex",
"Moderate osteoarthritis"
"Extensive lytic osseous metastatic disease",
"Multiple lytic lesions in the bones",
"Bilateral proximal humeri involvement",
"Increasing inflammatory changes",
"Intertrochanteric region",
"Infiltrative and destructive lesion of the proximal right femur",
"Associated soft tissue mass",
"Prior malignancy",
"Metastatic disease",
"Possibility of a primary osseous neoplasm or lymphoma cannot be excluded",
"The patient is at significant risk for pathologic fracture given the degree of bony involvement",
"Aggressive lytic metastasis",
"Femoral diaphysis",
"Cortical destruction",
"Permeation of the remainder of the cortex",
"Patient at high risk for impending pathologic fracture",
"Recommend urgent Orthopedic Oncologic consultation to assess for need for prophylactic fixation",
"Intramedullary",
"Enhancing lesion",
"Metastatic disease",
"Osseous metastatic lesions",
"Cortical thinning",
"Pathologic fracture",
"Endosteal scalloping",
"Intertrochanteric region",
"Tarlov cysts",
"Large lesion within the mid to distal humeral shaft",
"Extensive enhancement of the lesion",
"Enhancing extrasseous soft tissue component",
"Small areas of cortical discontinuity",
"Findings are highly concerning for primary or metastatic osseous lesion",
"Early lesion within the proximal humeral metaphysis",
"Infiltrative aggressive process",
"Pathological fracture",
"Necrotic appearing pelvic lymph node",
"Osteopenia",
"Degenerative findings",
"Large soft tissue mass",
"Intertrochanteric region of the right femur",
"Marked destruction of both the greater and lesser trochanter",
"Metastatic lesion involving the proximal right femur",
"Soft tissue component",
"Additional metastatic lesion involving the upper right iliac bone",
"Malignant marrow signal involving the proximal right femur",
"Soft tissue component",
"Diffuse osseous metastatic disease",
"Involving the axial and appendicular skeleton",
"Index lesion in the right femoral neck with SUV max = 8.2",
"Uptake within the posterior right elbow",
"Osseous lesion",
"Suspicious lesion",
"Parosteal in location",
"Redemonstrated diffuse lytic and sclerotic lesions in the spine, sternum, and ribs",
"Pathologic fractures of T3 and T4 with vertebral plana of T4",
"Compression fractures of T8 and T12 with similar degree of height loss and bony retropulsion",
"L2 compression deformity with interval increase in height loss",
"Redemonstrated left second rib lesion",
"Lytic lesion",
"Proximal femur",
"Sclerotic margins",
"Metastatic lytic lesion",
"Risk of pathologic fracture",
"Multifocal metastatic lesions",
"Hypermetabolic lesion",
"Sclerotic changes",
"Hypermetabolic lesion in the L5 vertebral body",
"Mid humeral diaphysis",
"Mildly displaced pathologic fracture",
"Lytic lesion",
"Soft tissue swelling",
"Nodules in the lung",
"Proximal left femur",
"Femoral neck",
"Lytic lesions",
"Cortical destruction",
"Extraosseous soft tissue",
"Pathologic fracture risk",
"Metastatic disease",
"Progression of metastatic disease",
"Proximal right humerus",
"Pathologic fracture",
"Permeative bone destruction",
"Metastatic breast cancer",
"Risk of further pathological fractures",
"Multiple lytic lesions in the skeleton",
"Proximal right femur",
"Lytic lesions",
"Hepatic and osseous metastatic lesions",
"Diffuse osseous metastatic disease",
"Sclerotic lesions in multiple midthoracic vertebral bodies",
"Pathologic fracture risk",
"Femoral head",
"Femoral neck",
"Proximal femur",
"Multiple metastatic lesions in the femur and pelvis",
"Ill-defined sclerosis",
"Additional focus of sclerosis with some intermixed lucency",
"Medial tibial metaphysis",
"Expansile lytic lesion involving the left anterior 5th rib",
"Lytic lesion within the L2 vertebral body",
"Lytic lesion within the left anterior acetabulum",
"Mildly displaced subacute fracture of the left femur",
"Surrounding callus",
"Degenerative change within the spine",
"Multiple lytic lesions",
"Pathologic fracture of the right femoral neck",
"Lytic lesion involving the femoral head",
"Lytic lesions within the right iliac bone and acetabulum",
"Lytic lesions within the right sacrum, the left posterior iliac bone, S1",
"Lytic lesion within the posterior elements of L1 with encroachment into the spinal canal",
"Lytic lesion eroding the body of T3 on the right side with extension into the spinal canal",
"Acute nondisplaced pathologic fracture of the medial femoral condyle",
"Aggressive appearing",
"T1 hypointense, T2 hyperintense",
"Lytic lesion",
"Satellite lesions in the distal femur and proximal tibia",
"Metastatic prostate carcinoma, given history",
"Fracture",
"Femoral metaphysis",
"Intercondylar notch",
"Possibility of a renal cell metastatic lesion within the left humeral head",
"Possibility of metastatic disease to the left humeral head",
"Incompletely imaged metastatic lesion within the C6 vertebral body",
"Similar appearance of mixed sclerotic and lytic lesion within the T8 vertebral body",
"Sclerosis of the right aspect of the T10 vertebral body extending to the lamina, pedicle, and transverse process, which was previously lytic",
"Similar mixed lytic and sclerotic lesions of the posterior right second and eighth ribs and anterior left sixth rib",
"Mixed sclerotic and lytic appearance of the lesions",
"Innumerable osseous metastatic lesions",
"Lytic lesion",
"Aggressive appearing",
"Marrow replacing lesions",
"Humeral neck",
"Concerning for metastatic disease",
"ill-defined",
"Expansile",
"Heterogeneously T2 hyperintense, T1 hypointense",
"Surrounding marrow edema",
"Aggressive",
"Permeative lesion",
"Compatible with neoplasm",
"Differential diagnosis includes primary tumor, sarcoma, possible metastatic deposit or plasmacytoma",
"Consider MRI",
"Enlarging lytic lesion of the right femoral head",
"Unchanged sclerotic lesions of L1 and the left ilium",
"Possible metastasis",
"Lytic lesion",
"Sclerotic lesion",
"Bone metastasis",
"Lytic lesion",
"Soft tissue mass",
"Associated with history of renal cell cancer",
"Possible metastases",
"Lytic metastatic bone lesion",
"Cortical erosion",
"Cortical breakthrough",
"Localized pathologic fracture",
"Mixed lytic/sclerotic lesions",
"Proximal humerus diaphysis",
"Metastatic",
"Lucent lesion in the femur",
"Suspicious",
"Bone scan",
"Lytic bone lesion in the left femur",
"Concerning for a bone tumor",
"Cortical disruption",
"Aggressive process",
"High risk for pathologic fracture",
"Additional smaller lytic lesion",
"Permeative lesion in the right proximal humeral shaft",
"Surrounding periosteal reaction",
"Concerning for metastatic disease",
"Bone reaction",
"Bone metastases",
"Pathologic fracture without significant displacement",
"Lytic lesion of the lesser trochanter of the right proximal femur",
"Presence of lytic lesions and sclerotic lesions indicating bone metastasis",
"Clinical history of bone mets and right femur pain",
"Presence of multiple osseous sclerotic and lytic lesions",
"Small foci of lytic lesions at the proximal right femur",
"Hypermetabolic osseous lesions",
"Multifocal",
"Axial and proximal appendicular osseous metastases",
"Sarcoma not excluded",
"Osteolytic lesions"
    
]
 
# --- NLTK DATA DOWNLOAD ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    print("NLTK data (punkt, wordnet) found.")
except LookupError:
    print("Downloading necessary NLTK data (punkt, wordnet)...")
    nltk.download('punkt')
    nltk.download('wordnet')
    print("Download complete.")
 
 
# --- PHASE 1: PRE-PROCESSING ---
def preprocess_terms(term_list):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed_list = []
    for term in term_list:
        term = term.lower()
        term = re.sub(r'[^\w\s]', '', term)
        tokens = nltk.word_tokenize(term)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        processed_list.append(" ".join(lemmatized_tokens))
    return processed_list
 


 
# --- PHASE 2: LEXICAL (SET-BASED) ANALYSIS ---
def perform_lexical_analysis(doctors_terms, llm_terms):
    # Suppress the specific deprecation warning from matplotlib-venn
    warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib_venn')
 
    doctors_set = set(doctors_terms)
    llm_set = set(llm_terms)
    intersection = doctors_set.intersection(llm_set)
    doctors_specific = doctors_set.difference(llm_set)
    llm_specific = llm_set.difference(doctors_set)
 
    print("\n--- PHASE 2: LEXICAL ANALYSIS RESULTS ---")
    print(f"\nTotal unique terms from Doctors: {len(doctors_set)}")
    print(f"Total unique terms from LLM: {len(llm_set)}")
    print(f"\n[+] Shared Terminology ({len(intersection)} terms):")
    print(intersection if intersection else "None")
    print(f"\n[!] Clinician-Only Terms (Missed by LLM) ({len(doctors_specific)} terms):")
    print(doctors_specific if doctors_specific else "None")
    print(f"\n[?] LLM-Only Terms (Not in Clinician List) ({len(llm_specific)} terms):")
    print(llm_specific if llm_specific else "None")
 
    # --- Venn Diagram Customization ---
    plt.figure(figsize=(10, 8)) # You can adjust the overall figure size here
 
    # Create the Venn diagram and store the output object
    v = venn2(subsets=(len(doctors_specific), len(llm_specific), len(intersection)),
              set_labels=('Clinician Terminology', 'LLM Terminology'),
              # 1. Customize Colors
              set_colors=('skyblue', '#e7274f'), # Change the circle colors
              alpha=0.8 # Set the transparency
             )
 
    # 2. Customize Font and Size of Set Labels ('Clinician Terminology', etc.)
    for text in v.set_labels:
        if text:  # Check if the label exists
            text.set_fontsize(16)
            text.set_fontname('serif')
 
    # 3. Customize Font and Size of Subset Labels (the numbers inside)
    for text in v.subset_labels:
        if text:  # Check if a subset label exists
            text.set_fontsize(18)
            text.set_fontweight('bold')
            text.set_color('white')
 
    # plt.title('Lexical Overlap of Medical Terminologies', fontsize=16, fontname='Times New Roman')
    venn_filename = "venn_diagram_comparison.png"
    plt.savefig(venn_filename, dpi=600, bbox_inches='tight')
    print(f"\n[+] Venn diagram saved to '{venn_filename}'")
    plt.show()
 
    # Restore default warning behavior if needed elsewhere in your code
    warnings.resetwarnings()
 
    return doctors_specific, llm_specific

 
# --- PHASE 3: SEMANTIC AND CONCEPTUAL ANALYSIS ---
 
def generate_heatmap_by_similarity_ranges(similarity_matrix, doc_terms, llm_terms):
    """
    Identifies 30 semantically similar terms across specified score ranges
    and generates a heatmap for just those pairs with specific font settings.
    """
    print("\n--- Generating Heatmap by Similarity Score Ranges ---")
    # Flatten the similarity matrix to a list of (llm_term, doc_term, score) tuples
    flat_scores = []
    for i, llm_term in enumerate(llm_terms):
        for j, doc_term in enumerate(doc_terms):
            flat_scores.append((llm_term, doc_term, similarity_matrix[i, j]))
 
    # Sort all scores in descending order to prioritize higher similarity within each bin
    flat_scores.sort(key=lambda x: x[2], reverse=True)
 
    # Define the similarity score ranges and the number of terms to select from each
    ranges = {
        "0.9-0.95": (0.9, 0.95, 10),
        "0.85-0.9": (0.85, 0.9, 10),
        "0.8-0.85": (0.8, 0.85, 10),
    }

 
    selected_terms = []
    # Use a set to keep track of already selected term pairs to avoid duplicates
    selected_pairs = set()
 
    # Iterate through the defined ranges to select the terms
    for name, (lower_bound, upper_bound, count) in ranges.items():
        # Filter terms that fall within the current range
        range_specific_terms = [
            item for item in flat_scores
            if lower_bound <= item[2] < upper_bound
        ]
 
        # Add the top 'count' terms from this range that haven't been selected yet
        added_count = 0
        for term_pair in range_specific_terms:
            if added_count >= count:
                break
            # Create a unique key for the pair to check for duplicates
            pair_key = tuple(sorted((term_pair[0], term_pair[1])))
            if pair_key not in selected_pairs:
                selected_terms.append(term_pair)
                selected_pairs.add(pair_key)
                added_count += 1
        print(f"Found {added_count} terms in the {name} range.")
 
    if not selected_terms:
        print("No term pairs found within the specified similarity ranges.")
        return
    # Create a DataFrame from the selected terms and pivot for the heatmap
    heatmap_df = pd.DataFrame(selected_terms, columns=['LLM_Term', 'Clinician_Term', 'Similarity'])
    pivot_data = heatmap_df.pivot(index='LLM_Term', columns='Clinician_Term', values='Similarity')
 
    # --- Font and Plotting Customization ---
    # Set the global font to serif
    plt.rcParams['font.family'] = 'serif'
    # Generate the heatmap
    plt.figure(figsize=(28, 16))
    ax = sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap="turbo", 
        fmt=".3f", # Updated to round scores to two decimal places
        linewidths=0.9, 
        cbar_kws={'label': ''},
        annot_kws={"size": 16, "family": "serif"} # Set annotation font and size
    )

    # --- Get the color bar and set the label with a specific font size ---
    cbar = ax.collections[0].colorbar
    cbar.set_label('Cosine Similarity', size=24) # Adjust size=14 as needed
    
    # --- NEW: Increase the font size of the color bar's tick labels ---
    cbar.ax.tick_params(labelsize=16) # Adjust size=12 as needed
 

    # Set title and labels with the specified font and size
    # plt.title(
    #     f'Heatmap of {len(selected_terms)} Semantically Similar Terms in Defined Ranges', 
    #     fontsize=18, 
    #     fontname='serif'
    # )
    plt.xlabel('Clinician-Only Terms', fontsize=26, fontname='serif')
    plt.ylabel('LLM-Only Terms', fontsize=26, fontname='serif')
    # Set tick labels with the specified font and size
    plt.xticks(rotation=45, ha='right', fontsize=20, fontname='serif')
    plt.yticks(rotation=0, fontsize=20, fontname='serif')


   # --- LOGIC TO WRAP LONG TICK LABELS ---
    WORD_LIMIT = 8 # Set the word count threshold
 
    # Wrap X-tick labels
    new_xticklabels = []
    # ax.get_xticklabels() returns a list of Matplotlib Text objects
    for tick in ax.get_xticklabels():
        text = tick.get_text()
        words = text.split()
        if len(words) > WORD_LIMIT:
            # Find the midpoint to split the text
            mid_point = len(words) // 2
            # Join the first half and the second half with a newline
            new_text = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
            new_xticklabels.append(new_text)
        else:
            new_xticklabels.append(text)
    # Set the new, potentially multi-line labels
    ax.set_xticklabels(new_xticklabels)
 
    # Wrap Y-tick labels
    new_yticklabels = []
    for tick in ax.get_yticklabels():
        text = tick.get_text()
        words = text.split()
        if len(words) > WORD_LIMIT:
            mid_point = len(words) // 2
            new_text = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
            new_yticklabels.append(new_text)
        else:
            new_yticklabels.append(text)
    # Set the new, potentially multi-line labels
    ax.set_yticklabels(new_yticklabels)
    plt.tight_layout()


    # Get the x-tick locations
    xticks_locs = ax.get_xticks()
 
    # Get limits for drawing lines
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
 
    # Draw a vertical line at each x-tick position
    for xloc in xticks_locs:
        ax.vlines(
            x=xloc,
            ymin=ymin,
            ymax=ymax,
            colors='gray',     # Or any other color you prefer
            linestyles='--',  # Dashed line style
            alpha=0.1         # Adjust for more or less fading (0.0 to 1.0)
        )
    
    # Draw horizontal lines from each y-tick position
    yticks_locs = ax.get_yticks()
    for yloc in yticks_locs:
        ax.hlines(
            y=yloc,
            xmin=xmin,
            xmax=xmax,
            colors='gray',     # Or any other color you prefer
            linestyles='-',  # Dashed line style
            alpha=0.1         # Adjust for more or less fading (0.0 to 1.0)
        )

    heatmap_filename = "ranged_similarity_heatmap.png"
    plt.savefig(heatmap_filename, dpi=600, bbox_inches='tight')
    print(f"[+] Ranged heatmap saved to '{heatmap_filename}'")
    plt.show()
 
 
def perform_semantic_analysis(doctors_specific, llm_specific):
    print("\n--- PHASE 3: SEMANTIC SIMILARITY ANALYSIS ---")
    if not llm_specific or not doctors_specific:
        print("One of the specific term lists is empty. Skipping semantic analysis.")
        return
 
    model_name = '.../BioBert Model'
    print(f"\nLoading model '{model_name}'. This may take a moment...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load model. Please ensure you have an internet connection. Error: {e}")
        return
 
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.numpy()
 
    print("Generating embeddings for all specific terms...")
    llm_embeddings = np.array([get_embedding(term) for term in llm_specific])
    doc_embeddings = np.array([get_embedding(term) for term in doctors_specific])
    similarity_matrix = cosine_similarity(llm_embeddings, doc_embeddings)
 
    # --- Generate a readable heatmap for terms in specified similarity ranges ---
    generate_heatmap_by_similarity_ranges(similarity_matrix, list(doctors_specific), list(llm_specific))
 
    print("\nAnalyzing similarities...")
    results = []
    for i, llm_term in enumerate(llm_specific):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_idx]
        best_doc_term = list(doctors_specific)[best_match_idx]
        if best_score > 0.9: category = "High (Likely Synonym)"
        elif best_score > 0.75: category = "Moderate (Related Concept)"
        else: category = "Low (Likely Unrelated)"
        results.append({
            "LLM-Only Term": llm_term,
            "Closest Clinician Term": best_doc_term,
            "Similarity Score": f"{best_score:.2f}",
            "Analysis": category
        })
    results_df = pd.DataFrame(results)
    # print("\n[+] Similarity analysis for LLM-Only terms:")
    # print(results_df.to_string())
 
    # --- SAVE THE RESULTS TO A CSV FILE ---
    csv_filename = "llm_similarity_analysis.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n[+] Full similarity analysis saved to '{csv_filename}'")
 
# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Starting terminology analysis pipeline...")
    doctors_clean = preprocess_terms(doctors_list)
    llm_clean = preprocess_terms(llm_list)
    doctors_specific, llm_specific = perform_lexical_analysis(doctors_clean, llm_clean)
    perform_semantic_analysis(doctors_specific, llm_specific)
    all_clean_terms = set(doctors_clean + llm_clean)
    print("\nAnalysis complete.")
