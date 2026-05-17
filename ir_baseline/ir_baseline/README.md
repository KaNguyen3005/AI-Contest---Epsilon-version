# IR Baselines — Truy Xuất Tài Liệu Khoa Học

> **Yêu cầu**: Không dùng pretrained model. Toàn bộ code thuần Python (stdlib + math).

---

## Cấu trúc project

```
ir_baseline/
├── run_all_baselines.py        # ← Điểm chạy chính
├── baselines/
│   ├── baseline1_tfidf.py      # TF-IDF + Cosine Similarity
│   ├── baseline2_bm25.py       # BM25 (Okapi BM25)
│   └── baseline3_bm25_prf.py   # BM25 + Pseudo-Relevance Feedback
└── utils/
    ├── preprocessing.py        # Tokenize, stop-word removal, Porter Stemmer
    ├── data_loader.py          # Load corpus / queries / answers / save CSV
    └── evaluate.py             # Precision, Recall, F1
```

---

## Chuẩn bị dữ liệu

```
data/
├── docs/
│   ├── 1.txt
│   ├── 2.txt
│   └── ... (1400 files)
├── public_test_queries.csv
└── public_test_answers.csv     (chỉ dùng khi evaluate)
```

---

## Cách chạy

### Chạy tất cả baselines (có evaluate)
```bash
python run_all_baselines.py \
    --corpus  data/docs \
    --queries data/public_test_queries.csv \
    --answers data/public_test_answers.csv \
    --output  submissions/
```

### Chạy vòng bí mật (không có answers)
```bash
python run_all_baselines.py \
    --corpus  data/docs \
    --queries data/private_test_queries.csv \
    --top_k   20 \
    --output  submissions/
```

### Chạy riêng từng baseline
```bash
cd ir_baseline
python -m baselines.baseline1_tfidf
python -m baselines.baseline2_bm25
python -m baselines.baseline3_bm25_prf
```

---

## Các Baseline

### Baseline 1 — TF-IDF + Cosine Similarity
| Thành phần | Chi tiết |
|---|---|
| Term weighting | Augmented TF × IDF (sklearn-style smoothing) |
| Similarity | Cosine similarity |
| Stemming | Porter Stemmer (built-in, no NLTK) |
| Stop-words | 174 English stopwords (hard-coded) |

**Khi nào dùng**: Baseline đơn giản nhất, chạy nhanh. Dùng để so sánh cơ sở.

---

### Baseline 2 — BM25 (Okapi BM25)
| Thành phần | Chi tiết |
|---|---|
| k1 | 1.5 (TF saturation) |
| b | 0.75 (length normalisation) |
| IDF | Robertson IDF: log((N - df + 0.5) / (df + 0.5) + 1) |

**Khi nào dùng**: Thường là baseline mạnh nhất trong class bag-of-words. Nên dùng làm primary baseline.

---

### Baseline 3 — BM25 + Pseudo-Relevance Feedback (Rocchio)
| Thành phần | Chi tiết |
|---|---|
| Pass 1 | BM25 → top-5 pseudo-relevant docs |
| Expansion | Top-10 terms từ pseudo-relevant docs |
| Rocchio α | 1.0 (query weight) |
| Rocchio β | 0.8 (feedback weight) |
| Pass 2 | BM25 với expanded query |

**Khi nào dùng**: Khi muốn cải thiện recall. Đặc biệt hiệu quả cho queries ngắn.

---

## Pipeline Xử Lý Văn Bản

```
Raw text
   │
   ▼
Lowercase + Remove punctuation
   │
   ▼
Tokenize (whitespace split)
   │
   ▼
Remove stopwords (174 English stopwords)
   │
   ▼
Porter Stemming (tự implement, không dùng NLTK)
   │
   ▼
[term1, term2, ...]
```

---

## Điều chỉnh Hyperparameters

### Top-K (quan trọng nhất với F1)
Script tự grid-search top-K nếu có `--answers`. Thông thường:
- F1 tốt nhất ở top-K ≈ số relevant docs trung bình mỗi query
- Tập Cranfield thường ~5–10 relevant docs/query → thử k ∈ {5,10,15,20}

### BM25
- `k1 ∈ [1.2, 2.0]`: Tăng nếu corpus có nhiều technical terms
- `b ∈ [0.5, 0.9]`: Giảm nếu doc length variance thấp

### PRF
- `prf_top_docs ∈ [3, 10]`: Tăng để có nhiều feedback hơn (rủi ro query drift)
- `prf_top_terms ∈ [5, 20]`: Số từ thêm vào query
- `beta ∈ [0.5, 1.0]`: Tăng để phụ thuộc nhiều hơn vào feedback

---

## Kết quả kỳ vọng (Cranfield dataset)

| Model | F1 (est.) |
|---|---|
| TF-IDF Cosine | ~0.35–0.45 |
| BM25 | ~0.45–0.55 |
| BM25 + PRF | ~0.50–0.60 |

*Số liệu ước tính, phụ thuộc vào top-K và preprocessing.*

---

## Hướng cải thiện tiếp theo (không pretrained)

1. **Query expansion với WordNet** — thêm synonyms vào query
2. **Bi-gram / n-gram indexing** — "heat conduction" vs "heat" + "conduction"
3. **Field-weighted BM25** — title vs abstract có trọng số khác nhau
4. **RM3 / KL-divergence language model** — thay Rocchio bằng language model PRF
5. **Score fusion** — kết hợp TF-IDF score + BM25 score (linear interpolation)
