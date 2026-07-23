"""HAIP - Content-derived golden set generator.

The original golden set was hand-written from document TITLES, giving ~0.53
lexical overlap between question and title. That makes retrieval trivially
easy (near keyword matching) and inflates every metric.

This script generates questions from document BODY TEXT instead, with an
explicit instruction not to reuse title vocabulary — producing questions
phrased the way a clinician would actually ask them.

Output: data/processed/qa_ground_truth_v2.json  (same schema as v1)
Usage:  python -m src.evaluation.build_golden_set
"""
from __future__ import annotations
import json, os, pickle, random, re, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv("/home/soorya/Documents/capstone/HAIP/healthcare-project/.env")

V1 = "data/processed/qa_ground_truth.json"
CHUNKS = "data/processed/chunks.pkl"
OUT = "data/processed/qa_ground_truth_v2.json"
QUESTIONS_PER_DOC = 2
MODEL = os.getenv("GOLDEN_SET_MODEL", "llama-3.1-8b-instant")

PROMPT = """You are helping build an evaluation set for a clinical guideline
retrieval system.

Below is text from a medical guideline document. Write {n} questions that a
doctor or hospital analyst would realistically ask, which this passage answers.

STRICT RULES:
- Do NOT reuse the drug name, procedure name, or title wording in the question.
  Describe the clinical situation instead.
  BAD:  "How is hyperkalaemia treated with sodium zirconium cyclosilicate?"
  GOOD: "What treatment options exist for a patient with elevated potassium?"
- Write how a clinician speaks, not how a document is titled.
- Each question must be answerable from this passage alone.
- Return ONLY a JSON array of question strings. No preamble, no markdown.

PASSAGE:
{passage}
"""


def main():
    v1 = json.load(open(V1))
    gold_ids = [d for q in v1["questions"] for d in q["relevant_doc_ids"]]

    chunks = pickle.load(open(CHUNKS, "rb"))
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    llm = ChatGroq(model=MODEL, temperature=0.3,
                   groq_api_key=os.environ["GROQ_API_KEY"])

    out, qid = [], 1
    for i, doc_id in enumerate(gold_ids, 1):
        docs = by_doc.get(doc_id, [])
        if not docs:
            print(f"  [{i}/{len(gold_ids)}] SKIP {doc_id[:8]} (no chunks)")
            continue
        # sample body text from the middle of the doc, avoiding title-heavy start
        body = " ".join(c["text"] for c in docs[1:4])[:1800] or docs[0]["text"][:1800]
        title = str(docs[0].get("title"))

        try:
            raw = llm.invoke(PROMPT.format(n=QUESTIONS_PER_DOC, passage=body)).content
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            questions = json.loads(raw)
        except Exception as e:
            print(f"  [{i}/{len(gold_ids)}] FAIL {doc_id[:8]}: {str(e)[:80]}")
            continue

        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue
            out.append({
                "id": f"v2_q{qid:02d}",
                "question": q.strip(),
                "relevant_doc_ids": [doc_id],
                "topic": title[:60] if title != "None" else "unknown",
                "source_title": title,
            })
            qid += 1
        print(f"  [{i}/{len(gold_ids)}] {doc_id[:8]} -> {len(questions)} questions")

    payload = {
        "description": ("Content-derived golden set for HAIP RAG evaluation. "
                        "Questions generated from document BODY TEXT by an LLM "
                        "with an explicit instruction not to reuse title "
                        "vocabulary, to avoid the lexical-overlap bias present "
                        "in v1 (mean title overlap 0.53)."),
        "generator_model": MODEL,
        "questions": out,
    }
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"\nGenerated {len(out)} questions -> {OUT}")


if __name__ == "__main__":
    main()
