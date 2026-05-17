"""
Data loading utilities.
"""

import os
import csv
from typing import Dict, List, Tuple


def load_corpus(corpus_dir: str) -> Dict[int, str]:
    """
    Load all {docID}.txt files from corpus_dir.
    Returns {doc_id: raw_text}.
    """
    corpus = {}
    for fname in os.listdir(corpus_dir):
        if fname.endswith(".txt"):
            try:
                doc_id = int(fname.replace(".txt", ""))
                with open(os.path.join(corpus_dir, fname), "r",
                          encoding="utf-8", errors="ignore") as f:
                    corpus[doc_id] = f.read()
            except ValueError:
                pass
    return corpus


def load_queries(query_csv: str) -> Dict[int, str]:
    """
    Load public_test_queries.csv → {query_id: query_text}.
    """
    queries = {}
    with open(query_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("query_id") or "").strip()
            query = (row.get("query") or "").strip()
            if not qid or not query:
                continue
            queries[int(qid)] = query
    return queries


def load_answers(answer_csv: str) -> Dict[int, List[int]]:
    """
    Load public_test_answers.csv → {query_id: [doc_id, ...]}.
    """
    answers = {}
    with open(answer_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid_text = (row.get("query_id") or "").strip()
            if not qid_text:
                continue
            qid = int(qid_text)
            doc_ids = [int(x) for x in (row.get("relevant_docIDs") or "").split()
                       if x.strip()]
            answers[qid] = doc_ids
    return answers


def save_submission(results: Dict[int, List[int]], output_path: str):
    """
    Save results as nlp_submission.csv.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "relevant_docIDs"])
        for qid in sorted(results.keys()):
            doc_ids = results[qid]
            writer.writerow([qid, " ".join(map(str, doc_ids))])
    print(f"Saved submission -> {output_path}")
