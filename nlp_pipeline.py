"""
nlp_pipeline.py
Core functions:
- extract_entities: Rule-based + spaCy patterns for Symptoms, Diagnosis, Treatment, Prognosis
- extract_keywords: KeyBERT
- sentiment_intent_for_utterance: transformer sentiment pipeline + simple intent rules
- structured_summary_from_transcript: build final JSON
- generate_soap_from_transcript: rule-based SOAP note
"""

import re
from typing import List, Dict, Any
import spacy
from spacy.matcher import PhraseMatcher
from keybert import KeyBERT
from transformers import pipeline

# Load spaCy model (ensure en_core_web_sm installed)
nlp = spacy.load("en_core_web_sm")

# Initialize KeyBERT (this will download a small embedding model if missing)
kw_model = KeyBERT(model="distilbert-base-nli-mean-tokens")

# Sentiment pipeline (HuggingFace). Uses a general sentiment model — ok for prototype.
sentiment_pipeline = pipeline("sentiment-analysis")


# Medical phrase lists for quick prototyping. Expand as needed.
MEDICAL_SYMPTOMS = [
    "neck pain",
    "back pain",
    "head impact",
    "headache",
    "stiffness",
    "backache",
    "pain",
    "nausea",
    "dizziness",
    "anxiety",
]

MEDICAL_DIAGNOSES = [
    "whiplash",
    "whiplash injury",
    "lower back strain",
    "concussion",
    "sprain",
]

MEDICAL_TREATMENTS = [
    "physiotherapy",
    "physiotherapy sessions",
    "painkillers",
    "analgesics",
    "rest",
    "ice",
    "heat therapy",
]

PROGNOSIS_TERMS = [
    "full recovery",
    "recovery expected",
    "no long-term damage",
    "no long term damage",
    "improving",
]


def _make_phrase_matcher(nlp, term_list: List[str], label: str):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in term_list]
    matcher.add(label, patterns)
    return matcher


SYM_MATCHER = _make_phrase_matcher(nlp, MEDICAL_SYMPTOMS, "SYMPTOM")
DIAG_MATCHER = _make_phrase_matcher(nlp, MEDICAL_DIAGNOSES, "DIAGNOSIS")
TREAT_MATCHER = _make_phrase_matcher(nlp, MEDICAL_TREATMENTS, "TREATMENT")
PROG_MATCHER = _make_phrase_matcher(nlp, PROGNOSIS_TERMS, "PROGNOSIS")


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract Symptoms, Diagnosis, Treatment, Prognosis using simple pattern matching and spaCy.
    Returns lists (deduplicated).
    """
    doc = nlp(text)
    symptoms = set()
    diagnoses = set()
    treatments = set()
    prognosis = set()

    # Phrase matchers
    for match_id, start, end in SYM_MATCHER(doc):
        span = doc[start:end].text
        symptoms.add(span)
    for match_id, start, end in DIAG_MATCHER(doc):
        span = doc[start:end].text
        diagnoses.add(span)
    for match_id, start, end in TREAT_MATCHER(doc):
        span = doc[start:end].text
        treatments.add(span)
    for match_id, start, end in PROG_MATCHER(doc):
        span = doc[start:end].text
        prognosis.add(span)

    # Numeric / session extraction (e.g., "ten physiotherapy sessions" or "10 sessions")
    session_matches = re.findall(r'(\b\d+\b)\s+(physiotherapy|sessions|session)', text, flags=re.I)
    for num, _ in session_matches:
        treatments.add(f"{num} physiotherapy sessions")
    # common words like "ten"
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for w, n in word_numbers.items():
        regex = rf'(\b{w}\b)\s+(physiotherapy|sessions|session)'
        if re.search(regex, text, flags=re.I):
            treatments.add(f"{n} physiotherapy sessions")

    # Heuristic: pain mentions as symptoms if "pain" appears with body part
    pain_matches = re.findall(r'(\b(neck|back|head|lower back|shoulder|arm|leg)\b).{0,20}\b(pain|ache|aching)\b',
                              text, flags=re.I)
    for m in pain_matches:
        body = m[0]
        symptoms.add(f"{body} pain")

    # Also pick up "hit my head" -> head impact
    if re.search(r'\bhit my head\b', text, flags=re.I) or re.search(r'\bhead on the steering wheel\b', text, flags=re.I):
        symptoms.add("head impact")

    return {
        "Symptoms": sorted(symptoms),
        "Diagnosis": sorted(diagnoses),
        "Treatment": sorted(treatments),
        "Prognosis": sorted(prognosis),
    }


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """
    Extract keywords using KeyBERT.
    """
    try:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
        # keywords returns list of (kw, score)
        return [kw for kw, score in keywords]
    except Exception as e:
        # fallback simple freq-based keywords
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = {}
        for w in words:
            if w in ("the", "and", "you", "patient", "doctor", "i", "a"):
                continue
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]


def sentiment_intent_for_utterance(utterance: str) -> Dict[str, Any]:
    """
    Return sentiment (Anxious / Neutral / Reassured) and intent (Seeking reassurance / Reporting symptoms / Expressing concern / Other).
    Uses a simple mapping over HF sentiment pipeline + pattern rules for intent.
    """
    # Basic sentiment mapping
    try:
        res = sentiment_pipeline(utterance[:512])[0]  # Truncate long text
        label = res['label']
        score = float(res.get('score', 0.0))
    except Exception:
        label = "NEUTRAL"
        score = 0.0

    # Map HF labels to our categories:
    # HF often returns POSITIVE/NEGATIVE. We'll interpret NEGATIVE -> Anxious (if patient text), POSITIVE -> Reassured.
    if label.upper() == "NEGATIVE":
        sentiment = "Anxious"
    elif label.upper() == "POSITIVE":
        # positive does not always mean "Reassured", so be conservative:
        sentiment = "Reassured" if score > 0.9 else "Neutral"
    else:
        sentiment = "Neutral"

    # Intent rules (simple)
    u = utterance.lower()
    if any(w in u for w in ["worried", "worry", "anxious", "concerned", "nervous", "scared", "afraid"]):
        intent = "Seeking reassurance"
    elif any(w in u for w in ["how long", "will i", "affect me", "future", "long-term", "long term", "worried about"]):
        intent = "Seeking reassurance"
    elif any(w in u for w in ["pain", "hurt", "injury", "accident", "stiffness", "ache", "backache"]):
        intent = "Reporting symptoms"
    elif any(w in u for w in ["thank", "thanks", "appreciate"]):
        intent = "Expressing gratitude"
    else:
        intent = "Other"

    return {"Sentiment": sentiment, "Score": score, "Intent": intent}


def parse_transcript_to_utterances(transcript: str) -> List[Dict[str, str]]:
    """
    Simple parser expecting lines prefixed with 'Doctor:' or 'Patient:' (case-insensitive).
    Returns list of dicts: {"speaker": "Doctor"|"Patient"|"Other", "text": "..."}
    """
    utterances = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(Doctor|Physician|Dr|Patient)\s*:\s*(.*)$', line, flags=re.I)
        if m:
            speaker = m.group(1).capitalize()
            text = m.group(2).strip()
            utterances.append({"speaker": speaker, "text": text})
        else:
            # continue last speaker if present, else consider Other
            if utterances:
                utterances[-1]["text"] += " " + line
            else:
                utterances.append({"speaker": "Other", "text": line})
    return utterances


def structured_summary_from_transcript(transcript: str) -> Dict[str, Any]:
    """
    Build a structured JSON summary for the patient from the transcript.
    Fields: Patient_Name (if found), Symptoms, Diagnosis, Treatment, Current_Status, Prognosis
    """
    utterances = parse_transcript_to_utterances(transcript)
    full_text = " ".join([u["text"] for u in utterances])
    entities = extract_entities(full_text)
    keywords = extract_keywords(full_text, top_n=10)

    # Attempt to capture patient name (Person entity)
    patient_name = None
    doc = nlp(full_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            patient_name = ent.text
            break

    # Current status: heuristics - last patient utterance mentioning "now", "currently", "occasion"
    current_status = None
    for u in reversed(utterances):
        if u["speaker"].lower() == "patient":
            t = u["text"].lower()
            if any(w in t for w in ["now", "currently", "occasion", "sometimes", "still"]):
                current_status = u["text"]
                break

    # Prognosis heuristic from mentions
    prognosis = entities.get("Prognosis", [])
    if not prognosis:
        # Look for "full recovery within X" pattern
        m = re.search(r'full recovery.*?within\s+(\w+\s*\w*)', full_text, flags=re.I)
        if m:
            prognosis = [f"Full recovery expected within {m.group(1)}"]

    result = {
        "Patient_Name": patient_name or "",
        "Symptoms": entities.get("Symptoms", []),
        "Diagnosis": entities.get("Diagnosis", []),
        "Treatment": entities.get("Treatment", []),
        "Current_Status": current_status or "",
        "Prognosis": prognosis,
        "Keywords": keywords,
    }
    return result


def generate_soap_from_transcript(transcript: str) -> Dict[str, Any]:
    """
    Rule-based SOAP note generation.
    """
    utterances = parse_transcript_to_utterances(transcript)
    full_text = " ".join([u["text"] for u in utterances])
    entities = extract_entities(full_text)

    # Subjective: gather chief complaint and HPI from patient utterances
    chief = ""
    hpi_sentences = []
    for u in utterances:
        if u["speaker"].lower() == "patient":
            text = u["text"]
            # If patient mentions accident or pain early on, mark CC
            if not chief and any(w in text.lower() for w in ["pain", "accident", "hurt", "ache", "whiplash"]):
                # take first short clause as chief complaint
                chief = text.split(".")[0]
            hpi_sentences.append(text)

    subjective = {
        "Chief_Complaint": chief or ", ".join(entities.get("Symptoms", [])),
        "History_of_Present_Illness": " ".join(hpi_sentences)
    }

    # Objective: look for physician exam statements or "physical examination" markers
    physical_exam = ""
    observations = ""
    for u in utterances:
        if u["speaker"].lower() in ("doctor", "physician", "dr"):
            t = u["text"].lower()
            if "physical examination" in t or "everything looks" in t or "range of motion" in t:
                physical_exam += u["text"] + " "
            # collect observations
            if "no tenderness" in t or "no signs" in t or "full range" in t:
                observations += u["text"] + " "

    if not physical_exam:
        # fallback: extract sentences mentioning range of movement, tenderness
        m = re.search(r'(full range of movement.*?\.?)', full_text, flags=re.I)
        if m:
            physical_exam = m.group(1)

    if not observations and physical_exam:
        observations = physical_exam

    objective = {
        "Physical_Exam": physical_exam.strip(),
        "Observations": observations.strip()
    }

    # Assessment: diagnosis mapping + severity heuristic
    diagnoses = entities.get("Diagnosis", [])
    severity = "Mild"
    if "whiplash" in " ".join([d.lower() for d in diagnoses]):
        severity = "Mild to Moderate"
    if any(w in full_text.lower() for w in ["severe", "hospital", "fracture", "surgery"]):
        severity = "Severe"

    assessment = {
        "Diagnosis": diagnoses,
        "Severity": severity
    }

    # Plan: suggest continuation of therapy if treatment present; else conservative advice
    plan_treat = []
    follow_up = "Return if symptoms worsen or persist."
    if entities.get("Treatment"):
        plan_treat = entities.get("Treatment")
        plan_treat.append("Use analgesics as needed")
    else:
        plan_treat = ["Conservative management (analgesics, rest)."]

    plan = {
        "Treatment": plan_treat,
        "Follow_Up": follow_up
    }

    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }

# For convenience: one-shot analyze
def analyze_transcript(transcript: str) -> Dict[str, Any]:
    utterances = parse_transcript_to_utterances(transcript)
    patient_utts = [u for u in utterances if u["speaker"].lower() == "patient"]
    sentiment_intent = []
    for u in patient_utts:
        si = sentiment_intent_for_utterance(u["text"])
        sentiment_intent.append({"utterance": u["text"], **si})

    structured = structured_summary_from_transcript(transcript)
    soap = generate_soap_from_transcript(transcript)
    entities = extract_entities(transcript)
    keywords = extract_keywords(transcript, top_n=10)

    return {
        "entities": entities,
        "keywords": keywords,
        "sentiment_intent": sentiment_intent,
        "structured_summary": structured,
        "soap": soap
    }

if __name__ == "__main__":
    # quick test
    sample = """Doctor: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Doctor: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Doctor: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Doctor: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Doctor: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
Doctor: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Doctor: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Doctor: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Doctor: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Doctor: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
Doctor: [Physical Examination Conducted]
Doctor: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That’s a relief!
Doctor: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Doctor: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Doctor: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.
"""
    out = analyze_transcript(sample)
    import json
    print(json.dumps(out, indent=2))
