
# Medical Note ICD Code Inference

This project practices Retrieval-Augmented Generation (RAG) using LangChain and OpenAI's GPT-3.5-turbo. The system processes medical notes to extract key information and map it to appropriate ICD codes using the RAG approach.

## Introduction


### Language Models (LLM)

A language model (LLM) is a type of artificial intelligence model designed to understand and generate human language. In this project, we use OpenAI’s GPT-3.5-turbo, which is a state-of-the-art LLM capable of understanding and generating human-like text based on the input it receives.

### Retrieval-Augmented Generation (RAG)

RAG is a technique that combines the strengths of retrieval-based models and generation-based models. In a RAG setup:

	•	The retrieval component fetches relevant documents or context from a knowledge base or vector store.
	•	The generation component (usually an LLM) then uses this retrieved context to generate more accurate and contextually appropriate responses or inferences.

This approach enhances the model’s ability to generate responses that are not only coherent but also factually grounded.

### LangChain

LangChain is a framework designed to simplify the development of applications using large language models. It provides tools for:

	•	Creating and managing prompts
	•	Chaining together multiple components such as retrievers, LLMs, and output parsers
	•	Handling inputs and outputs efficiently

In this project, LangChain is used to create chains that:

	•	Extract the ‘Assessment’ section from doctor’s notes
	•	Infer ICD codes based on the extracted assessment

LangChain’s integration with vector stores and its support for retrieval-augmented generation makes it a powerful tool for building complex NLP applications.

## Project Structure

	•	marag_icdcoding.py: Main script to run the project.
	•	utils.py: Utility functions for parsing ICD-10 descriptions.
	•	icd10cm_order_2024_top100.txt: Sample ICD-10 descriptions file.
	•	.env: Environment file containing API keys.

## Functions

load_dfdata(df)

Loads data from a DataFrame and returns ICD descriptions.

create_vectorstore(doc)

Creates a vector store from the provided documents using OpenAI embeddings.

extract_assessment(llm, anote)

Extracts the ‘Assessment’ section from a doctor’s note using a language model.

extract_demo(llm, vdb)

Extracts demographic information (name and age) from a doctor’s note.

extract_icd(llm, vdb)

Infers the related ICD codes from the content of a doctor’s note.

## Example Usage

Example Doctor’s Note

Here is a sample doctor’s note for testing:

note = """
Patient Name: John Doe
Date of Visit: June 17, 2024

Chief Complaint: High fever, abdominal pain, and diarrhea.

History of Present Illness:
John Doe, 45, reports a high fever for 5 days, abdominal cramps, and watery diarrhea. Also complains of fatigue and body aches.

Review of Systems:
    • Fever, chills, fatigue
    • Abdominal pain, diarrhea
    • No cough, shortness of breath, chest pain

Physical Examination:
    • Temp: 39.5°C, BP: 120/80, HR: 110, RR: 18
    • Appears ill, febrile
    • Abdominal tenderness, hyperactive bowel sounds

Assessment:
    Suspected typhoid fever
    Possible acute amebic dysentery
    Calicivirus enteritis

Plan:
    • Tests: CBC, blood cultures, stool analysis, Widal test, PCR for viruses
    • Treatment: Rehydration, Ciprofloxacin, antipyretics, isolation
    • Follow-Up: Return in 48 hours for results review

Signature:
Dr. Jane Smith, MD
Internal Medicine
"""

### Extract the assessment from the note
assessment = extract_assessment(llm, note)
print("Extracted Assessment:", assessment)

### Infer ICD codes from the extracted assessment
icd_codes = infer_icd_codes(assessment)
print("Inferred ICD Codes:", icd_codes)
