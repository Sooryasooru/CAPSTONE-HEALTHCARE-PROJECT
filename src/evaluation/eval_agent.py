"""HAIP - Agent evaluation: tool-selection accuracy, call validity, completion, steps."""
from __future__ import annotations
import csv, sys, json
sys.path.insert(0, ".")
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import build_agent

OUT = "data/eval/agent_results.csv"

# question -> tool(s) the agent SHOULD call
CASES = [
    ("What does the guideline say about managing delirium in hospital?", {"search_guidelines"}),
    ("How should acute heart failure be treated?", {"search_guidelines"}),
    ("What is recommended for migraine prevention?", {"search_guidelines"}),
    ("Show me the current hospital KPIs.", {"get_hospital_kpis"}),
    ("What is our readmission rate?", {"get_hospital_kpis"}),
    ("How many total encounters do we have?", {"get_hospital_kpis"}),
    ("Forecast admissions for the next 6 months.", {"forecast_admissions"}),
    ("What will admissions look like over the next 3 months?", {"forecast_admissions"}),
    ("Project our patient volume going forward.", {"forecast_admissions"}),
    ("How many doctors work in cardiology?", {"get_doctor_stats"}),
    ("Give me statistics on our doctors.", {"get_doctor_stats"}),
    ("Which departments have the most providers?", {"get_doctor_stats"}),
    ("What are our KPIs and what do the guidelines say about readmissions?",
     {"get_hospital_kpis", "search_guidelines"}),
    ("Forecast admissions and tell me our current encounter count.",
     {"forecast_admissions", "get_hospital_kpis"}),
    ("Hello, who are you?", set()),  # should answer directly, no tool
]

VALID_TOOLS = {"search_guidelines", "get_hospital_kpis",
               "forecast_admissions", "get_doctor_stats"}


def main():
    agent = build_agent()
    rows = []
    for i, (q, expected) in enumerate(CASES, 1):
        try:
            result = agent.invoke({"messages": [HumanMessage(content=q)]})
            msgs = result["messages"]
            called, invalid = [], 0
            for m in msgs:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        called.append(tc["name"])
                        if tc["name"] not in VALID_TOOLS:
                            invalid += 1
            called_set = set(called)
            correct = 1 if called_set == expected else 0
            # partial credit: did it call at least the right tools?
            partial = 1 if expected and expected <= called_set else correct
            completed = 1 if isinstance(msgs[-1], AIMessage) and msgs[-1].content.strip() else 0
            steps = len(msgs)
            err = ""
        except Exception as e:
            called_set, correct, partial, completed, steps, invalid = set(), 0, 0, 0, 0, 0
            err = str(e)[:120]

        rows.append({"n": i, "question": q,
                     "expected": "|".join(sorted(expected)) or "(none)",
                     "called": "|".join(sorted(called_set)) or "(none)",
                     "exact_match": correct, "partial_match": partial,
                     "invalid_calls": invalid, "completed": completed,
                     "steps": steps, "error": err})
        print(f"  [{i}/{len(CASES)}] {'OK ' if correct else 'MISS'} "
              f"exp={sorted(expected) or '-'} got={sorted(called_set) or '-'}")

    n = len(rows)
    acc = sum(r["exact_match"] for r in rows) / n
    part = sum(r["partial_match"] for r in rows) / n
    comp = sum(r["completed"] for r in rows) / n
    inval = sum(r["invalid_calls"] for r in rows)
    avg_steps = sum(r["steps"] for r in rows) / n

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 55)
    print(f"AGENT EVALUATION  (n={n})")
    print("=" * 55)
    print(f"  Tool-Selection Accuracy (exact) : {acc*100:.1f}%")
    print(f"  Tool-Selection (partial credit) : {part*100:.1f}%")
    print(f"  Task Completion Rate            : {comp*100:.1f}%")
    print(f"  Invalid / Hallucinated Calls    : {inval}")
    print(f"  Avg Steps per Query             : {avg_steps:.1f}")
    print("=" * 55)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
