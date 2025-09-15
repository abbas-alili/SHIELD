# SHIELD
SHIELD: An AI Framework for Skeletal Health Intelligence and Early Lesion Detection to Improve Orthopedic Referrals

The overall goal of this study was to introduce SHIELD (Skeletal Health Intelligence and Early Lesion Detection), an AI framework designed to prevent late referrals by providing highly accurate classifications and offering appropriate explainability to aid medical experts in their decision-making.  Our study had three aims. The primary aim of this work was to 1) develop an AI framework to classify radiology reports into three distinct referral categories: no referral needed, referral recommended, and referral/high-risk. This was achieved by fine-tuning the RadBERT-RoBERTa-4m17 model for MBD referral classification using a decade of data (January 2014 – May 2025) from two affiliated academic medical centers. RadBERT-RoBERTa-4m17 is a transformer-based language model developed for radiology and clinical NLP. A secondary aim was to 2) provide explanations for the framework's classification decisions, thereby increasing trust and adoption among medical providers. We utilized the Llama-3.1-8B-Instruct, Meta’s instruction-tuned language model that was released in July 2024, to accomplish this task. Our final aim was to 3) validate the explainability of the proposed framework and assess the overlap between the terminology used by the AI and a predefined set of terms established by orthopedic experts.

The SHIELD framework is composed of four main stages, each implemented in a dedicated Python script:
1. Model Training and Classification (SlidingWindowFineTune3Class_5CV.py)
This is the core of the framework where a classification model is trained to analyze radiology reports. Due to the lengthy nature of these reports, a sliding window technique is employed to process the text in manageable chunks. The model is trained using a 5-fold cross-validation strategy to ensure its robustness and generalizability. The output of this stage is a trained model capable of classifying new radiology reports into one of the three predefined categories.
2. Explainability with LLaMA 3 (runLLama3.py)
Once a report is classified, this script utilizes the powerful LLaMA 3 large language model to generate a detailed explanation for the given classification. The model is prompted to identify and articulate the specific medical terminology and findings within the report that support the classification, such as the presence of lytic lesions or the risk of a pathologic fracture.
3. Validation of Explainability (terminComparison.py)
To ensure the clinical relevance of the AI-generated explanations, this script performs a comparative analysis between the terminology used by the LLM and a predefined list of terms provided by clinicians. This validation is conducted through:
Lexical Analysis: Identifying the overlap and differences in terminology using set-based methods.
Semantic Analysis: Employing BioBERT, a language model specialized in biomedical text, to assess the conceptual similarity between the clinician's and the LLM's terms.
4. Performance Visualization (confMat_ROC_Generator.py)
This script is responsible for generating key performance metrics and visualizations for the classification model. It produces confusion matrices to show the model's accuracy for each class and ROC curves to illustrate the model's diagnostic ability at various threshold settings.

Key Features
3-Class Classification: Stratifies orthopedic oncology referrals to help manage patient care more effectively.
Explainable AI (XAI): Provides transparency in the AI's decision-making process, which is crucial for clinical adoption.
Clinical Terminology Validation: Uses BioBERT to ensure that the AI's explanations are medically sound and relevant.
Robust Evaluation: Employs 5-fold cross-validation to build a reliable and well-vetted classification model.

File Descriptions
SlidingWindowFineTune3Class_5CV.py: The main script for training the 3-class classification model using a sliding window approach and 5-fold cross-validation.
runLLama3.py: Uses a fine-tuned LLaMA 3 model to generate explanations for the classifications made by the model.
terminComparison.py: Compares the terminology from the LLM's explanations with a list of clinician-provided terms to validate the explainability.
confMat_ROC_Generator.py: Generates and saves the confusion matrices and ROC curves for the classification model.

Getting Started

Prerequisites
Python 3.7+
PyTorch
Transformers
pandas
scikit-learn
seaborn
matplotlib
nltk
tqdm
Installation
Clone the repository:
codeBash
 
