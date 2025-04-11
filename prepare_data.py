# step2_prepare_data.py
import os
import sys
sys.path.append('/Users/aaddharbhaduri/miniforge3/envs/env_2/lib/python3.12/site-packages')
from PyPDF2 import PdfReader

import json

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            texts.append(text.strip())
    return texts

# Simulate 10 PDFs with distinct content
def simulate_10_pdfs():
    pdf_contents = [
        "PDF 1: Function-Coherent Gambles with Non-Additive Sequential Dynamics",
        "PDF 2: Markets for Models",
        "PDF 3: Walrasian equilibrium: An alternate proof of existence and lattice structure",
        "PDF 4: How manipulable are prediction markets?",
        "PDF 5: Exchange Rate Sensitivity in Free Zone Trade: An Empirical Study of the Istanbul Atatürk Airport Free Zone",
        "PDF 6: The Limits of Search Algorithms",
        "PDF 7: When Should we Expect Non-Decreasing Returns from Data in Prediction Tasks?",
        "PDF 8: A Linear Theory of Multi-Winner Voting",
        "PDF 9: The Fragility of Sparsity",
        "PDF 10: Optimal transmission expansion modestly reduces decarbonization costs of U.S. electricity"
    ]
    return pdf_contents

# Generate 50 examples (5 per PDF)
def create_50_examples(pdf_texts):
    dataset = []
    for i, pdf_text in enumerate(pdf_texts, 1):
        # Split text into parts and create 5 examples per PDF
        sentences = pdf_text.split(". ")
        for j in range(5):
            if j < len(sentences):
                example = f"PDF {i}, Example {j+1}: {sentences[j]}."
            else:
                # Repeat last sentence with variation if fewer than 5
                example = f"PDF {i}, Example {j+1}: {sentences[-1]} (Additional note {j- len(sentences) + 2})."
            dataset.append({"text": example})
    return dataset

# Main execution
pdf_folder = "/Users/aaddharbhaduri/Downloads/docs"
os.makedirs(pdf_folder, exist_ok=True)

# Try to extract from PDFs; if empty, simulate data
pdf_texts = extract_text_from_pdfs(pdf_folder)
if not pdf_texts:
    print("No PDFs found. Simulating 10 PDFs and creating 50 examples.")
    pdf_texts = simulate_10_pdfs()
else:
    print(f"Extracted text from {len(pdf_texts)} PDFs.")
    if len(pdf_texts) < 10:
        print("Fewer than 10 PDFs found. Padding with simulated data.")
        pdf_texts.extend(simulate_10_pdfs()[len(pdf_texts):10])

# Ensure exactly 10 PDFs
pdf_texts = pdf_texts[:10] if len(pdf_texts) > 10 else pdf_texts + simulate_10_pdfs()[len(pdf_texts):10]


# # Create dataset with 50 examples
# dataset = create_50_examples(pdf_texts)

# # Save dataset
# with open("dataset.json", "w") as f:
#     json.dump(dataset, f)

# print("Dataset with 50 examples from 10 PDFs saved as dataset.json.")