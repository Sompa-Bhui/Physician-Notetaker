# Physician Notetaker (Rule-based Prototype)

This project extracts important medical information from a doctor–patient conversation and stores it in structured JSON format. The outputs include sentiment & intent JSON, a medical summary JSON, and a SOAP note JSON.

## Project Overview

- Uses a plain-text doctor–patient conversation as input.
- Applies a simple rule-based NLP approach to:
  - Detect patient **sentiment** (Anxious / Reassured / Neutral).
  - Detect patient **intent** (Expressing concern / Seeking reassurance / Reporting symptoms).
- From the extracted information it generates:
  - A medical summary JSON.
  - A standard SOAP (Subjective, Objective, Assessment, Plan) note JSON.

## How to Run

1. Install Python 3 and Jupyter Notebook.
2. Clone or download this repository to your local machine.
3. Open the `physician_notetaker.ipynb` notebook.
4. Run all cells from top to bottom:
   - The cell that defines the conversation text.
   - The cells that define and run the sentiment & intent analysis functions.
   - The cell that generates the medical summary JSON.
   - The cell that generates the SOAP note JSON.
5. The JSON outputs for each step will be printed inside the notebook.

## Handling Missing or Ambiguous Data

- If some information is not clearly mentioned in the transcript (for example exact diagnosis, number of sessions, or exact dates), the JSON can use values like `"Unknown"` or `null`.
- An additional field such as `"Needs_Review": true` can be used to indicate that a clinician should manually review that part.
- This prototype is meant to assist the documentation process, not to replace real clinical decision making.

## NLP Approach and Future Improvements

- Current implementation:
  - Simple keyword-based sentiment lists (positive and negative words).
  - Manually defined fields and templates to build the summary and SOAP note.
- Possible future improvements:
  - Use transformer-based models (such as BART or T5) for better automatic summarization.
  - Use medical language models (e.g., ClinicalBERT or medical NER models) to automatically extract diagnoses, symptoms, and treatments.
  - Train machine‑learning based sentiment and intent classifiers on annotated medical conversation datasets.

## Files in This Repository

- `physician_notetaker.ipynb` – main Jupyter notebook containing the full implementation.
- `README.md` – this documentation file describing the project, usage steps, and design choices.
