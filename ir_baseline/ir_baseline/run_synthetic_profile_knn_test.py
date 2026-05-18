"""
Evaluate ProfileKNN on synthetic queries/answers.

This script is a stress test, not a final submission runner. It checks whether
the profile-based retriever still behaves sensibly when labels come from a
separate synthetic query set, and verifies that query_id collisions do not
force predictions to use the colliding id's answer.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from baselines.baseline2_bm25 import BM25Retriever
from baselines.baseline7_profile_knn import ProfileKNNRetriever
from baselines.baseline8_adaptive_ensemble import AdaptiveProfileBM25Ensemble
from run_all_baselines import save_per_query_report
from utils.data_loader import load_answers, load_corpus, load_queries, save_submission
from utils.evaluate import evaluate, evaluate_per_query


def ensure_synthetic_files(query_path: str, answer_path: str):
    if os.path.isfile(query_path) and os.path.isfile(answer_path):
        return
    script_path = os.path.join(os.path.dirname(__file__), "generate_synthetic_queries.py")
    subprocess.run([sys.executable, script_path], check=True)


def subset(mapping: Dict[int, object], qids: List[int]) -> Dict[int, object]:
    qid_set = set(qids)
    return {qid: value for qid, value in mapping.items() if qid in qid_set}


def evaluate_bm25(corpus, queries, answers, top_k: int):
    retriever = BM25Retriever(k1=1.5, b=0.75, stem=True)
    retriever.fit(corpus)
    results = retriever.retrieve_all(queries, top_k=top_k)
    return evaluate(results, answers, verbose=False), results


def evaluate_profile_loo(corpus, queries, answers, top_k: int, args):
    retriever = ProfileKNNRetriever(
        neighbors=args.neighbors,
        min_similarity=args.min_similarity,
        profile_weight=args.profile_weight,
        lexical_weight=args.lexical_weight,
    )
    retriever.fit(corpus, train_queries=queries, train_answers=answers)
    results = retriever.retrieve_all(queries, top_k=top_k, leave_one_out=True)
    return evaluate(results, answers, verbose=False), results


def evaluate_adaptive_ensemble_loo(corpus, queries, answers, top_k: int, args):
    retriever = AdaptiveProfileBM25Ensemble(
        neighbors=args.neighbors,
        min_similarity=args.min_similarity,
        disagreement_profile_weight=args.disagreement_profile_weight,
    )
    retriever.fit(corpus, train_queries=queries, train_answers=answers)
    results = retriever.retrieve_all(queries, top_k=top_k, leave_one_out=True)
    return evaluate(results, answers, verbose=False), results


def evaluate_blocked_holdout(corpus, queries, answers, top_k: int, folds: int, args):
    qids = sorted(queries)
    fold_metrics = []
    fold_results: Dict[int, List[int]] = {}

    for fold_idx in range(folds):
        test_qids = qids[fold_idx::folds]
        train_qids = [qid for qid in qids if qid not in set(test_qids)]

        train_queries = subset(queries, train_qids)
        train_answers = subset(answers, train_qids)
        test_queries = subset(queries, test_qids)
        test_answers = subset(answers, test_qids)

        retriever = ProfileKNNRetriever(
            neighbors=args.neighbors,
            min_similarity=args.min_similarity,
            profile_weight=args.profile_weight,
            lexical_weight=args.lexical_weight,
        )
        retriever.fit(corpus, train_queries=train_queries, train_answers=train_answers)
        results = retriever.retrieve_all(test_queries, top_k=top_k, leave_one_out=False)
        fold_results.update(results)
        fold_metrics.append(evaluate(results, test_answers, verbose=False))

    all_metrics = evaluate(fold_results, answers, verbose=False)
    return all_metrics, fold_metrics, fold_results


def overlap_count(predicted: List[int], relevant: List[int]) -> int:
    return len(set(predicted) & set(relevant))


def query_id_collision_check(corpus, queries, answers, top_k: int, args) -> Tuple[bool, Dict[str, object]]:
    qids = sorted(queries)
    if len(qids) < 2:
        return False, {"reason": "need at least two synthetic queries"}

    colliding_qid = qids[0]
    content_source_qid = qids[1]
    private_like_queries = {colliding_qid: queries[content_source_qid]}

    retriever = ProfileKNNRetriever(
        neighbors=args.neighbors,
        min_similarity=args.min_similarity,
        profile_weight=args.profile_weight,
        lexical_weight=args.lexical_weight,
    )
    retriever.fit(corpus, train_queries=queries, train_answers=answers)
    results = retriever.retrieve_all(private_like_queries, top_k=top_k, leave_one_out=False)
    predicted = results[colliding_qid]

    collision_answer_overlap = overlap_count(predicted, answers[colliding_qid])
    content_answer_overlap = overlap_count(predicted, answers[content_source_qid])
    passed = content_answer_overlap >= collision_answer_overlap

    return passed, {
        "colliding_query_id": colliding_qid,
        "content_source_query_id": content_source_qid,
        "predicted": predicted,
        "colliding_id_answer": answers[colliding_qid],
        "content_source_answer": answers[content_source_qid],
        "colliding_id_overlap": collision_answer_overlap,
        "content_source_overlap": content_answer_overlap,
    }


def print_metrics(label: str, metrics: Dict[str, float]):
    print(
        f"{label:28s} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/Cranfield")
    parser.add_argument("--queries", default="data/synthetic_queries.csv")
    parser.add_argument("--answers", default="data/synthetic_answers.csv")
    parser.add_argument("--output_dir", default="submissions/synthetic_profile_knn")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min_similarity", type=float, default=0.08)
    parser.add_argument("--profile_weight", type=float, default=0.80)
    parser.add_argument("--lexical_weight", type=float, default=0.20)
    parser.add_argument("--disagreement_profile_weight", type=float, default=0.20)
    args = parser.parse_args()

    ensure_synthetic_files(args.queries, args.answers)
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    answers = load_answers(args.answers)
    print(f"Synthetic queries: {len(queries)} | answers: {len(answers)} | top_k={args.top_k}")

    bm25_metrics, bm25_results = evaluate_bm25(corpus, queries, answers, args.top_k)
    loo_metrics, loo_results = evaluate_profile_loo(corpus, queries, answers, args.top_k, args)
    ensemble_metrics, ensemble_results = evaluate_adaptive_ensemble_loo(
        corpus, queries, answers, args.top_k, args
    )
    holdout_metrics, fold_metrics, holdout_results = evaluate_blocked_holdout(
        corpus, queries, answers, args.top_k, args.folds, args
    )

    print_metrics("BM25 synthetic", bm25_metrics)
    print_metrics("ProfileKNN LOO", loo_metrics)
    print_metrics("Adaptive ensemble LOO", ensemble_metrics)
    print_metrics("ProfileKNN holdout", holdout_metrics)
    for idx, metrics in enumerate(fold_metrics, start=1):
        print_metrics(f"  holdout fold {idx}", metrics)

    collision_passed, collision_info = query_id_collision_check(corpus, queries, answers, args.top_k, args)
    print("Query-id collision check:", "PASS" if collision_passed else "FAIL")
    for key, value in collision_info.items():
        print(f"  {key}: {value}")

    save_submission(bm25_results, os.path.join(args.output_dir, "bm25_synthetic_submission.csv"))
    save_submission(loo_results, os.path.join(args.output_dir, "profile_knn_loo_synthetic_submission.csv"))
    save_submission(ensemble_results, os.path.join(args.output_dir, "adaptive_ensemble_synthetic_submission.csv"))
    save_submission(holdout_results, os.path.join(args.output_dir, "profile_knn_holdout_synthetic_submission.csv"))

    save_per_query_report(
        evaluate_per_query(loo_results, answers),
        os.path.join(args.output_dir, "profile_knn_loo_synthetic_per_query.csv"),
    )
    save_per_query_report(
        evaluate_per_query(holdout_results, answers),
        os.path.join(args.output_dir, "profile_knn_holdout_synthetic_per_query.csv"),
    )
    save_per_query_report(
        evaluate_per_query(ensemble_results, answers),
        os.path.join(args.output_dir, "adaptive_ensemble_synthetic_per_query.csv"),
    )
    print(f"Synthetic reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
