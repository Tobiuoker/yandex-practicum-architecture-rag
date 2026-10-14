from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from task_4.rag import RAGService
from task_7.query_logger import append_jsonl
from task_7.query_logger import serialize_sources

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BASE_DIR.parent

GOLDEN_FILE = BASE_DIR / "golden_questions.txt"

LOG_FILE = PROJECT_ROOT / "logs/logs.jsonl"

SUMMARY_FILE = BASE_DIR / "summary.json"


RETRIEVAL_K = 3


def load_golden_questions() -> list[dict]:
    questions = []

    with GOLDEN_FILE.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                questions.append(json.loads(line))

    return questions


def retrieval_metrics(
    sources,
    expected_sources,
):
    if not expected_sources:
        return None, None

    expected = set(expected_sources)

    for rank, source in enumerate(
        sources[:RETRIEVAL_K],
        start=1,
    ):
        if source.source in expected:
            return True, 1 / rank

    return False, 0.0


def semantic_similarity(
    model,
    actual: str,
    expected: str,
) -> float:
    """
    Semantic similarity is an auxiliary heuristic.

    The same embedding model is reused to avoid loading
    another evaluator model.
    """

    embeddings = model.encode(
        [
            f"classification: {actual}",
            f"classification: {expected}",
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return float(
        np.dot(
            embeddings[0],
            embeddings[1],
        )
    )


def required_terms_ok(
    answer: str,
    terms: list[str],
) -> bool:
    answer = answer.casefold()

    return all(term.casefold() in answer for term in terms)


def evaluate_known(
    *,
    rag,
    test,
    answer,
    retrieved,
):
    hit, reciprocal_rank = retrieval_metrics(
        retrieved,
        test["expected_sources"],
    )

    similarity = semantic_similarity(
        rag.embedder.model,
        answer.text,
        test["expected_answer"],
    )

    terms_ok = required_terms_ok(
        answer.text,
        test.get(
            "required_terms",
            [],
        ),
    )

    answer_ok = answer.answered_from_kb and terms_ok

    passed = hit and answer_ok

    if not hit:
        failure_stage = "retrieval"

    elif not answer_ok:
        failure_stage = "generation"

    else:
        failure_stage = None

    return {
        "retrieval_hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "answer_similarity": round(similarity, 4),
        "required_terms_ok": terms_ok,
        "passed": passed,
        "failure_stage": failure_stage,
    }


def evaluate_unknown(
    answer,
):
    passed = not answer.answered_from_kb

    return {
        "retrieval_hit": None,
        "reciprocal_rank": None,
        "answer_similarity": None,
        "required_terms_ok": None,
        "passed": passed,
        "failure_stage": (None if passed else "abstention"),
    }


def save_log(record: dict) -> None:
    append_jsonl(LOG_FILE, record)


def run_test(
    rag,
    test,
):
    started = time.perf_counter()

    # Retrieval is measured independently from generation.
    retrieved = rag.retriever.retrieve(
        test["question"],
        top_k=RETRIEVAL_K,
    )

    answer = rag.ask(test["question"])

    latency_ms = int((time.perf_counter() - started) * 1000)

    if test["should_answer"]:
        metrics = evaluate_known(
            rag=rag,
            test=test,
            answer=answer,
            retrieved=retrieved,
        )

    else:
        metrics = evaluate_unknown(answer)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": test["id"],
        "topic": test["topic"],
        "query": test["question"],
        "answer": answer.text,
        "answer_length": len(answer.text),
        "expected_should_answer": test["should_answer"],
        "answered": answer.answered_from_kb,
        "chunks_found": len(retrieved),
        "top_similarity": (
            round(
                retrieved[0].similarity,
                4,
            )
            if retrieved
            else None
        ),
        "sources": serialize_sources(retrieved),
        "latency_ms": latency_ms,
        **metrics,
    }


def build_summary(records):
    known = [record for record in records if record["expected_should_answer"]]

    unknown = [record for record in records if not record["expected_should_answer"]]

    retrieval_hits = [record["retrieval_hit"] for record in known]

    reciprocal_ranks = [record["reciprocal_rank"] for record in known]

    similarities = [record["answer_similarity"] for record in known]

    topic_stats = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
        }
    )

    for record in records:
        topic = record["topic"]

        topic_stats[topic]["total"] += 1

        if record["passed"]:
            topic_stats[topic]["passed"] += 1

    for stats in topic_stats.values():
        stats["pass_rate"] = round(
            stats["passed"] / stats["total"],
            4,
        )

    summary = {
        "questions": len(records),
        "passed": sum(record["passed"] for record in records),
        "overall_pass_rate": round(
            sum(record["passed"] for record in records) / len(records),
            4,
        ),
        "retrieval_hit_at_3": round(
            sum(retrieval_hits) / len(retrieval_hits),
            4,
        )
        if retrieval_hits
        else None,
        "mrr_at_3": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks),
            4,
        )
        if reciprocal_ranks
        else None,
        "mean_answer_similarity": round(
            sum(similarities) / len(similarities),
            4,
        )
        if similarities
        else None,
        "correct_abstention_rate": round(
            sum(not item["answered"] for item in unknown) / len(unknown),
            4,
        )
        if unknown
        else None,
        "failures_by_stage": {
            "retrieval": sum(item["failure_stage"] == "retrieval" for item in records),
            "generation": sum(item["failure_stage"] == "generation" for item in records),
            "abstention": sum(item["failure_stage"] == "abstention" for item in records),
        },
        "topics": dict(topic_stats),
    }

    return summary


def main():
    rag = RAGService.from_env()

    tests = load_golden_questions()

    # Start every evaluation run with a clean result file.
    LOG_FILE.unlink(missing_ok=True)

    records = []

    for index, test in enumerate(
        tests,
        start=1,
    ):
        print(f"[{index}/{len(tests)}] " f"{test['question']}")

        record = run_test(
            rag,
            test,
        )

        records.append(record)

        save_log(record)

        print("PASS" if record["passed"] else ("FAIL: " f"{record['failure_stage']}"))

    summary = build_summary(records)

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    print(f"Questions: " f"{summary['questions']}")

    print(f"Passed: " f"{summary['passed']}")

    print("Hit@3: " f"{summary['retrieval_hit_at_3']}")

    print("MRR@3: " f"{summary['mrr_at_3']}")

    print("Correct abstention: " f"{summary['correct_abstention_rate']}")

    print("\nFailures:")

    for stage, count in summary["failures_by_stage"].items():
        print(f"  {stage}: {count}")

    print("\nTopics:")

    for topic, stats in summary["topics"].items():
        print(f"  {topic}: " f"{stats['passed']}/" f"{stats['total']} " f"({stats['pass_rate']:.0%})")


if __name__ == "__main__":
    main()
