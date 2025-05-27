from __future__ import annotations

from textwrap import dedent
from typing import List

SYSTEM_QA_GEN = dedent(
    """
    You are a military subject-matter expert writing a synthetic
    question-answer dataset for your boss.

    For **each passage** the user sends **(exactly one passage at a time)**:

      • Write **ONE** short, fact-based question whose answer is found
        *verbatim* in that passage. Avoid yes/no or vague opinion
        questions.
      • Output the answer **exactly as it appears** in the passage
        (case-preserving, no added words).
      • Return the result as a single-line JSON object with keys
        "question" and "answer".  Example:

        {"question":"What is the first step of TLP?","answer":"Receive the mission"}

    Do **not** repeat the passage text and do not add extra keys.
    """
).strip()

USER_QA_BATCH = dedent(
    """
    Passages (separated by \\n---\\n):

    {passages}

    Remember: one JSON line per passage, no additional commentary.
    """
)

SYSTEM_JUDGE = dedent(
    """
    You are an impartial grader.

    Given a *question*, the *reference answer* (gold) and a *candidate
    answer*:

      • Return **"1"** if the candidate conveys the same meaning as the
        reference answer (case, punctuation and minor wording differences
        do not matter).
      • Otherwise return **"0"**.

    Return **ONLY** the single character 1 or 0 — no additional text.
    """
).strip()


def build_judge_user(question: str, reference: str, candidate: str) -> str:
    return dedent(
        f"""
        Question: {question}
        Reference answer: {reference}
        Candidate answer: {candidate}
        Does the candidate match the reference? Return 1 or 0.
        """
    ).strip()


def build_rag_prompt(context: List[str], question: str) -> str:
    numbered = [f"[{i+1}] {c}" for i, c in enumerate(context)]
    ctx = "\n\n".join(numbered)

    return dedent(
        f"""
        ### Context
        {ctx}

        ### Instructions
        • Use only the numbered passages above.  
        • Cite passage numbers in brackets, e.g. [1] or [2, 3]. 
        • Bracket numbers would correspond to the retrieved passages. 
        • If the answer is missing, write exactly: **Not in context**

        ### Question
        {question}

        ### Answer (concise, with citations, cite passage numbers in brackets []):
        """
    ).lstrip()


PROMPT_REGISTRY = {
    "system_qa_gen": SYSTEM_QA_GEN,
    "system_judge": SYSTEM_JUDGE,
}
