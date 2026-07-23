"""HAIP - Generation evaluation (Faithfulness, Answer Relevancy) via RAGAS + Gemini."""
from __future__ import annotations

import sys, types
# --- shim: langchain-community 0.4.x removed chat_models.vertexai, ragas imports it
_m = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_m.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _m

import json, os, csv
sys.path.insert(0, ".")
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag.retriever import Retriever

load_dotenv()
GT = "data/processed/qa_ground_truth.json"
OUT = "data/eval/generation_results.csv"
CACHE = "data/eval/answers_cache.json"
K = 5

PROMPT = ("You are a careful medical-guideline assistant. Answer using ONLY "
      "the numbered context passages. If the answer is not in the context, "
      "say the guidelines do not cover it. Be concise and factual.\n\n"
      "Context passages:\n\n{ctx}\n\nQuestion: {q}")


def main():
    gt = json.load(open(GT))["questions"]
    r = Retriever(index_path="data/processed/faiss.index",
                  metadata_path="data/processed/chunks.pkl")
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=judge_model,
                   groq_api_key=os.environ["GROQ_API_KEY"],
                   temperature=0)


    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        questions, answers, contexts = c["questions"], c["answers"], c["contexts"]
        print(f"Loaded {len(questions)} cached answers -> skipping generation")
    else:
        questions, answers, contexts = _generate(gt, r, llm)
        json.dump({"questions": questions, "answers": answers, "contexts": contexts},
                  open(CACHE, "w"), indent=2)
        print(f"Cached answers -> {CACHE}")
    ds = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})
    result = evaluate(
        ds, metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        run_config=RunConfig(timeout=300, max_workers=2, max_retries=3),
    )
    df = result.to_pandas()
    df.to_csv(OUT, index=False)
    f_score = df["faithfulness"].mean()
    a_score = df["answer_relevancy"].mean()
    print("\n" + "=" * 55)
    print(f"GENERATION EVALUATION  (n={len(gt)}, k={K})")
    print("=" * 55)
    print(f"  Faithfulness       : {f_score:.3f}")
    print(f"  Answer Relevancy   : {a_score:.3f}")
    print(f"  Hallucination Rate : {1 - f_score:.3f}")
    print("=" * 55)
    print(f"Saved -> {OUT}")
def _generate(gt, r, llm):
    questions, answers, contexts = [], [], []
    for i, q in enumerate(gt, 1):
        hits = r.retrieve(q["question"], top_k_rerank=K)
        ctx = [h["text"] for h in hits]
        prompt = PROMPT.format(ctx="\n\n".join(f"[{j}] {t}" for j, t in enumerate(ctx, 1)),
                               q=q["question"])
        ans = llm.invoke(prompt).content
        questions.append(q["question"]); answers.append(ans); contexts.append(ctx)
        print(f"  [{i}/{len(gt)}] answered: {q['id']}")
    return questions, answers, contexts


if __name__ == "__main__":
    main()
