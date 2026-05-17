"""
Evaluation: Precision, Recall, F1 (macro-averaged over all queries).
Matches the competition scoring criterion.
"""

from typing import Dict, List, Tuple


def precision_recall_f1(
    predicted: List[int],
    relevant: List[int],
) -> Tuple[float, float, float]:
    pred_set = set(predicted)
    rel_set  = set(relevant)

    tp = len(pred_set & rel_set)
    p  = tp / len(pred_set) if pred_set else 0.0
    r  = tp / len(rel_set)  if rel_set  else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate(
    results:  Dict[int, List[int]],
    answers:  Dict[int, List[int]],
    verbose:  bool = True,
) -> Dict[str, float]:
    from typing import Tuple   # local to avoid circular import issues

    precisions, recalls, f1s = [], [], []

    for qid, relevant in answers.items():
        predicted = results.get(qid, [])
        p, r, f1 = precision_recall_f1(predicted, relevant)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p  = sum(precisions) / len(precisions)
    macro_r  = sum(recalls)    / len(recalls)
    macro_f1 = sum(f1s)        / len(f1s)

    metrics = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}

    if verbose:
        print(f"  Precision : {macro_p:.4f}")
        print(f"  Recall    : {macro_r:.4f}")
        print(f"  F1        : {macro_f1:.4f}")

    return metrics


def evaluate_per_query(
    results: Dict[int, List[int]],
    answers: Dict[int, List[int]],
) -> Dict[int, Dict[str, object]]:
    """Return per-query metrics and ranks of relevant docs for diagnostics."""
    report: Dict[int, Dict[str, object]] = {}
    for qid, relevant in answers.items():
        predicted = results.get(qid, [])
        p, r, f1 = precision_recall_f1(predicted, relevant)
        ranks = {
            doc_id: predicted.index(doc_id) + 1 if doc_id in predicted else None
            for doc_id in relevant
        }
        report[qid] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "relevant_ranks": ranks,
            "predicted": predicted,
        }
    return report
