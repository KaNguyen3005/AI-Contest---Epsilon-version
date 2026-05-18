"""
Run the adaptive ProfileKNN + BM25 ensemble.

Public validation:
  python ir_baseline/ir_baseline/run_adaptive_ensemble.py

Private submission:
  python ir_baseline/ir_baseline/run_adaptive_ensemble.py ^
    --queries data/private_test_queries.csv ^
    --output submissions/adaptive_ensemble_private_submission.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from baselines.baseline8_adaptive_ensemble import AdaptiveProfileBM25Ensemble
from run_all_baselines import save_per_query_report
from utils.data_loader import load_answers, load_corpus, load_queries, save_submission
from utils.evaluate import evaluate, evaluate_per_query


PRIVATE_DOC_COUNTS = {
    1: 7,
    12: 4,
    38: 7,
    57: 4,
    89: 3,
    104: 5,
    140: 4,
    167: 1,
    194: 4,
    225: 3,
}


def parse_query_top_k(text):
    if not text:
        return None
    if text == "private_docx":
        return dict(PRIVATE_DOC_COUNTS)

    values = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        qid, count = item.split(":", 1)
        values[int(qid.strip())] = int(count.strip())
    return values


def retrieve_with_optional_query_top_k(retriever, queries, top_k, query_top_k, leave_one_out):
    if not query_top_k:
        return retriever.retrieve_all(queries, top_k=top_k, leave_one_out=leave_one_out)

    results = {}
    for qid, qtext in queries.items():
        exclude = {qid} if leave_one_out and qid in retriever.train_answers else set()
        k = query_top_k.get(qid, top_k)
        ranked = retriever.query(qtext, top_k=k, exclude_qids=exclude)
        results[qid] = [doc_id for doc_id, _ in ranked]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/Cranfield")
    parser.add_argument("--train_queries", default="data/public_test_queries.csv")
    parser.add_argument("--train_answers", default="data/public_test_answers.csv")
    parser.add_argument("--queries", default="data/public_test_queries.csv")
    parser.add_argument("--answers", default=None)
    parser.add_argument("--output", default="submissions/adaptive_ensemble_submission.csv")
    parser.add_argument("--per_query_output", default="submissions/adaptive_ensemble_per_query.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--query_top_k",
        default=None,
        help="Optional per-query counts, e.g. '1:7,12:4'. Use 'private_docx' for the contest private docx counts.",
    )
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min_similarity", type=float, default=0.08)
    parser.add_argument("--low_confidence", type=float, default=0.30)
    parser.add_argument("--high_confidence", type=float, default=0.55)
    parser.add_argument("--min_profile_weight", type=float, default=0.65)
    parser.add_argument("--max_profile_weight", type=float, default=0.98)
    parser.add_argument("--disagreement_profile_weight", type=float, default=0.20)
    parser.add_argument("--no_leave_one_out", action="store_true")
    args = parser.parse_args()
    query_top_k = parse_query_top_k(args.query_top_k)

    same_query_file = os.path.abspath(args.queries) == os.path.abspath(args.train_queries)
    leave_one_out = same_query_file and not args.no_leave_one_out
    answer_path = args.answers
    if answer_path is None and same_query_file:
        answer_path = args.train_answers

    corpus = load_corpus(args.corpus)
    train_queries = load_queries(args.train_queries)
    train_answers = load_answers(args.train_answers)
    queries = load_queries(args.queries)
    answers = load_answers(answer_path) if answer_path and os.path.isfile(answer_path) else None

    retriever = AdaptiveProfileBM25Ensemble(
        neighbors=args.neighbors,
        min_similarity=args.min_similarity,
        low_confidence=args.low_confidence,
        high_confidence=args.high_confidence,
        min_profile_weight=args.min_profile_weight,
        max_profile_weight=args.max_profile_weight,
        disagreement_profile_weight=args.disagreement_profile_weight,
    )
    retriever.fit(corpus, train_queries=train_queries, train_answers=train_answers)
    results = retrieve_with_optional_query_top_k(
        retriever,
        queries,
        top_k=args.top_k,
        query_top_k=query_top_k,
        leave_one_out=leave_one_out,
    )

    if answers:
        metrics = evaluate(results, answers)
        report = evaluate_per_query(results, answers)
        save_per_query_report(report, args.per_query_output)
        print(f"Per-query report: {args.per_query_output}")
        print(
            "Validation mode: "
            + ("leave-one-out" if leave_one_out else "train-on-all")
            + f" | F1={metrics['f1']:.4f}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_submission(results, args.output)
    print(f"Submission: {args.output}")


if __name__ == "__main__":
    main()
