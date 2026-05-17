"""
Run the supervised profile-kNN baseline.

Public validation, using leave-one-out:
  python ir_baseline/ir_baseline/run_profile_knn_baseline.py

Private submission, training on public labels:
  python ir_baseline/ir_baseline/run_profile_knn_baseline.py ^
    --queries data/private_test_queries.csv ^
    --output submissions/profile_knn_submission.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from baselines.baseline7_profile_knn import ProfileKNNRetriever
from utils.data_loader import load_answers, load_corpus, load_queries, save_submission
from utils.evaluate import evaluate, evaluate_per_query
from run_all_baselines import save_per_query_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/Cranfield")
    parser.add_argument("--train_queries", default="data/public_test_queries.csv")
    parser.add_argument("--train_answers", default="data/public_test_answers.csv")
    parser.add_argument("--queries", default="data/public_test_queries.csv")
    parser.add_argument("--answers", default=None)
    parser.add_argument("--output", default="submissions/profile_knn_submission.csv")
    parser.add_argument("--per_query_output", default="submissions/profile_knn_per_query.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min_similarity", type=float, default=0.08)
    parser.add_argument("--profile_weight", type=float, default=0.80)
    parser.add_argument("--lexical_weight", type=float, default=0.20)
    parser.add_argument(
        "--no_leave_one_out",
        action="store_true",
        help="Disable public validation leave-one-out. Private inference does this automatically.",
    )
    args = parser.parse_args()

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

    retriever = ProfileKNNRetriever(
        neighbors=args.neighbors,
        min_similarity=args.min_similarity,
        profile_weight=args.profile_weight,
        lexical_weight=args.lexical_weight,
    )
    retriever.fit(corpus, train_queries=train_queries, train_answers=train_answers)
    results = retriever.retrieve_all(
        queries,
        top_k=args.top_k,
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
