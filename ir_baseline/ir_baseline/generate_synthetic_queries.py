"""
Generate synthetic queries and answers from Cranfield corpus.
Output: synthetic_queries.csv + synthetic_answers.csv
"""
import os, csv, re, math, random
from collections import defaultdict, Counter

random.seed(42)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "Cranfield")

# --- Load corpus ---
corpus = {}
for fname in os.listdir(CORPUS_DIR):
    if fname.endswith(".txt"):
        doc_id = int(fname.replace(".txt", ""))
        with open(os.path.join(CORPUS_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
            corpus[doc_id] = f.read().strip()
print(f"Loaded {len(corpus)} documents")

# --- Preprocessing ---
STOPWORDS = set("a an the is are was were be been being have has had do does did will would shall "
    "should may might can could of in to for on with at by from as into through about between after "
    "before above below up down out off over under again further then once here there when where why "
    "how all each every both few more most other some such no nor not only own same so than too very "
    "and but if or because until while".split())

def tokenize(text):
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

# --- Build TF-IDF ---
doc_tokens = {}
df = Counter()
for doc_id, text in corpus.items():
    tokens = tokenize(text)
    doc_tokens[doc_id] = tokens
    df.update(set(tokens))

N = len(corpus)
idf = {t: math.log(N / (c + 1)) for t, c in df.items()}

doc_keywords = {}
for doc_id, tokens in doc_tokens.items():
    tf = Counter(tokens)
    scored = [(t, tf[t] * idf.get(t, 0)) for t in set(tokens)]
    scored.sort(key=lambda x: -x[1])
    doc_keywords[doc_id] = scored[:10]

# --- Inverted index ---
term_to_docs = defaultdict(set)
for doc_id, tokens in doc_tokens.items():
    for t in set(tokens):
        term_to_docs[t].add(doc_id)

# --- Generate queries ---
templates = [
    "what are the effects of {0} on {1} in {2}",
    "how does {0} relate to {1} and {2}",
    "what is the relationship between {0} and {1} for {2}",
    "investigation of {0} and {1} in the context of {2}",
    "analysis of {0} with respect to {1} and {2}",
    "what methods are used for {0} analysis involving {1} and {2}",
    "how can {0} be applied to problems involving {1} and {2}",
    "what are the characteristics of {0} under {1} conditions with {2}",
    "theoretical study of {0} and its effect on {1} near {2}",
    "experimental results for {0} measurements on {1} at {2}",
]

queries = []
answers = []
used_seeds = set()
query_id = 1

doc_ids_list = sorted(corpus.keys())
random.shuffle(doc_ids_list)

for seed_doc in doc_ids_list:
    if len(queries) >= 100:
        break
    if seed_doc in used_seeds:
        continue

    keywords = doc_keywords[seed_doc]
    if len(keywords) < 4:
        continue

    top_terms = [t for t, _ in keywords[:6]]
    related_scores = Counter()
    for term in top_terms:
        for doc_id in term_to_docs.get(term, []):
            if doc_id != seed_doc:
                related_scores[doc_id] += 1

    related = [did for did, cnt in related_scores.items() if cnt >= 3]
    related.sort(key=lambda d: -related_scores[d])
    related = related[:3]

    relevant_docs = [seed_doc] + related

    n_terms = random.randint(3, 5)
    query_terms = [t for t, _ in keywords[:n_terms]]

    template = random.choice(templates)
    terms_for_query = query_terms[:3]
    while len(terms_for_query) < 3:
        terms_for_query.append(query_terms[0])
    query_text = template.format(*terms_for_query)

    if len(query_terms) > 3:
        query_text += f" considering {' '.join(query_terms[3:])}"

    queries.append((query_id, query_text))
    answers.append((query_id, relevant_docs))
    used_seeds.add(seed_doc)
    for d in related:
        used_seeds.add(d)
    query_id += 1

print(f"Generated {len(queries)} queries")

# --- Save ---
out_q = os.path.join(DATA_DIR, "synthetic_queries.csv")
out_a = os.path.join(DATA_DIR, "synthetic_answers.csv")

with open(out_q, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "query"])
    for qid, qtext in queries:
        w.writerow([qid, qtext])

with open(out_a, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "relevant_docIDs"])
    for qid, docs in answers:
        w.writerow([qid, " ".join(map(str, docs))])

print(f"Saved: {out_q}")
print(f"Saved: {out_a}")

print("\n--- Sample queries ---")
for qid, qtext in queries[:5]:
    rel = [d for q, d in answers if q == qid][0]
    print(f"  Q{qid}: {qtext}")
    print(f"    Relevant: {rel}")
