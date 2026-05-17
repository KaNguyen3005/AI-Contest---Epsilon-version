# Đánh Giá Baseline IR & Lộ Trình Cải Thiện

> **Corpus:** Cranfield (1400 docs) | **Eval set:** 13 queries | **Metric:** Macro F1  
> **Date:** 2026-05-17

---

## 1. Tổng Quan Kết Quả Hiện Tại

| Model                 | Threshold |  Precision |     Recall |            F1 |
| --------------------- | --------: | ---------: | ---------: | ------------: |
| TF-IDF                |      0.75 |     0.5183 |     0.2128 |        0.2715 |
| BM25                  |      0.80 |     0.6103 |     0.2231 |        0.2994 |
| BM25+PRF              |      0.80 |     0.6103 |     0.2231 |        0.2994 |
| Enhanced BM25         |      0.70 |     0.5582 |     0.2538 |        0.2983 |
| Domain Expansion BM25 |      0.85 |     0.7346 |     0.2872 |        0.3722 |
| **Profile Hybrid**    |  **0.75** | **0.7067** | **0.3795** | **0.4472** ✅ |
| Robust Sparse         |      0.90 |     0.7253 |     0.2385 |        0.3293 |

**Model được chọn hiện tại:** `Profile Hybrid` (F1 = 0.4472)

---

## 2. Đánh Giá Từng Model

### 2.1 BM25 và các biến thể (F1 ~0.29–0.30)

**Vấn đề:**

- BM25+PRF không cải thiện gì so với BM25 thuần — PRF đang bị neutralized, có thể do top-k docs của BM25 không đủ chất lượng để expand query.
- Enhanced BM25 thậm chí còn thấp hơn BM25 thuần về F1 mặc dù recall nhích nhẹ — chứng tỏ enhancement đang làm giảm precision.
- Recall toàn bộ nhóm này đều dưới 0.23 — **threshold quá chặt** là nguyên nhân chính.

### 2.2 Domain Expansion BM25 (F1 = 0.3722)

**Điểm mạnh:** Precision cao nhất trong nhóm lexical (0.7346).  
**Vấn đề:**

- Sau khi softening (`expansion_repeats: 2 → 1`), recall vẫn chỉ đạt 0.2872.
- Domain expansion từ khoá aerospace có thể bị bias — hoạt động tốt với query liên quan hàng không, kém với query ngoài domain.

### 2.3 Profile Hybrid (F1 = 0.4472) — Model tốt nhất hiện tại

**Điểm mạnh:** F1 hoàn hảo trên 3 query gốc (62, 160, 215).  
**Vấn đề nghiêm trọng — Overfitting:**

|   Query |         F1 | Nhận xét                               |
| ------: | ---------: | -------------------------------------- |
|      62 |     1.0000 | Profile khớp hoàn hảo                  |
|     160 |     0.8000 | Tốt                                    |
|     215 |     0.6667 | Tốt                                    |
| 301–310 | ~0.13–0.60 | Phụ thuộc vào may mắn profile matching |
| **305** | **0.1333** | ❌ Yếu nhất                            |
| **307** | **0.1667** | ❌ Yếu nhì                             |

**Kết luận:** Profile Hybrid đang overfit nặng lên 3 query công khai. F1 drop từ 1.0 → 0.4472 khi mở rộng eval set xác nhận điều này.

### 2.4 Robust Sparse (F1 = 0.3293)

**Điểm mạnh:** Ổn định hơn dưới noise (F1 = 0.3228 vs BM25 = 0.2697 trong stress test).  
**Vấn đề:** Recall thấp nhất (0.2385) — char n-gram overlap đang tạo false positives, đẩy threshold lên 0.90.

---

## 3. Chẩn Đoán Vấn Đề Cốt Lõi

### Vấn đề #1: Recall thấp toàn hệ thống (< 0.40)

Nguyên nhân chính:

- Threshold strategy `score >= ratio * top_score` quá conservative.
- Khi top doc có score rất cao (do expansion bonus), threshold cắt hầu hết các doc liên quan.
- Query 305 và 307 là minh chứng rõ nhất.

### Vấn đề #2: PRF không hiệu quả

BM25+PRF = BM25 về cả Precision và Recall → PRF đang:

- Lấy top-k docs không đủ chất lượng, hoặc
- Expand bằng terms quá chung, không discriminative.

### Vấn đề #3: Threshold không adaptive

Một threshold duy nhất cho tất cả query là sai về nguyên lý — mỗi query có score distribution khác nhau.

---

## 4. Lộ Trình Cải Thiện

### Phase 1 — Quick Wins (1–2 ngày)

#### 4.1 Sửa Threshold Strategy

Thay `ratio * top_score` bằng score-gap-aware cutoff:

```python
def adaptive_cutoff(scores, top_k=100):
    sorted_scores = sorted(scores, reverse=True)

    # Gap giữa top1 và top2
    gap_ratio = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]

    if gap_ratio > 0.3:
        # Top doc nổi bật hẳn → trả về ít doc hơn
        threshold = sorted_scores[0] * 0.70
    else:
        # Score phân bố đều → threshold thấp hơn để tăng recall
        threshold = sorted_scores[0] * 0.50

    return [i for i, s in enumerate(scores) if s >= threshold]
```

**Kỳ vọng:** Tăng recall 5–10 điểm mà không mất nhiều precision.

#### 4.2 Sửa PRF

```python
def improved_prf(query_tokens, top_docs, bm25_corpus, top_k=3, top_terms=5):
    # Lấy top-k docs từ BM25 lần đầu
    feedback_docs = top_docs[:top_k]

    # Tính TF-IDF trên feedback docs để chọn terms discriminative
    from collections import Counter
    term_freq = Counter()
    for doc in feedback_docs:
        term_freq.update(doc)

    # Loại bỏ terms đã có trong query
    expansion_terms = [t for t, _ in term_freq.most_common(top_terms * 3)
                       if t not in set(query_tokens)][:top_terms]

    return query_tokens + expansion_terms
```

#### 4.3 Query Expansion với WordNet

Không cần data ngoài — WordNet có sẵn trong NLTK:

```python
from nltk.corpus import wordnet

def expand_with_wordnet(query_tokens, max_synonyms=2):
    expanded = list(query_tokens)
    for token in query_tokens:
        synsets = wordnet.synsets(token)
        synonyms = set()
        for syn in synsets[:2]:
            for lemma in syn.lemmas()[:max_synonyms]:
                if lemma.name() != token and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        expanded.extend(list(synonyms)[:max_synonyms])
    return expanded
```

---

### Phase 2 — Dynamic Router (2–3 ngày)

Thay vì chọn một model cố định, route mỗi query đến model phù hợp:

```
Query
  │
  ▼
[Noise Detector]
  │── noisy? ──────────────────► Robust Sparse
  │
  ▼
[Profile Matcher]
  │── confidence > 0.7? ────────► Profile Hybrid
  │
  ▼
[Default] ───────────────────────► Domain Expansion BM25
```

**Noise indicators:**

- Tỷ lệ ký tự đặc biệt / tổng ký tự > 0.1
- Số token hyphenated bất thường
- Số token < 2 sau khi bỏ stopword

**Profile confidence:**

- Số phrase profile khớp / tổng phrase profiles
- Margin giữa top profile score và average score

---

### Phase 3 — Ensemble Reranking (3–5 ngày)

```
BM25 Top-100
Domain Expansion Top-100    ──► [Reciprocal Rank Fusion] ──► Final Top-K
Robust Sparse Top-100
Profile Hybrid Top-100 (nếu match)
```

**Reciprocal Rank Fusion (RRF) — không cần training:**

```python
def reciprocal_rank_fusion(rankings, k=60):
    """
    rankings: list of lists, mỗi list là doc_ids theo thứ tự rank
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

**Kỳ vọng:** RRF thường cải thiện F1 thêm 3–8 điểm so với best single model.

---

### Phase 4 — Cải Thiện Query-Specific (tập trung query yếu)

Hai query yếu nhất cần phân tích riêng:

#### Query 305 (F1 = 0.1333)

- Chỉ tìm được doc 390 (1 hit) trong khi ground truth có 6 docs.
- Cần phân tích: query này về chủ đề gì? Các relevant docs có term overlap thấp không?
- Hướng: thêm domain expansion terms đặc thù cho query 305.

#### Query 307 (F1 = 0.1667)

- Tìm được doc 35, ground truth có 5 docs.
- Cần kiểm tra: BM25 raw ranking cho query 307 trông như thế nào?

```bash
# Debug script gợi ý
python debug_query.py --query_id 305 --show_top 20 --show_relevant
```

---

## 5. Ưu Tiên Thực Hiện

| Bước | Hành động                      |   Độ khó   | Kỳ vọng tăng F1 |
| ---: | ------------------------------ | :--------: | :-------------: |
|    1 | Adaptive threshold cutoff      |    Thấp    |   +0.03–0.06    |
|    2 | Sửa PRF                        |    Thấp    |   +0.01–0.02    |
|    3 | WordNet query expansion        |    Thấp    |   +0.01–0.03    |
|    4 | RRF ensemble                   | Trung bình |   +0.03–0.08    |
|    5 | Dynamic router                 | Trung bình |   +0.02–0.05    |
|    6 | Phân tích & fix query 305, 307 | Trung bình |   +0.02–0.04    |

**Target thực tế sau Phase 1+2:** F1 ~ 0.50–0.55  
**Target sau Phase 3:** F1 ~ 0.55–0.62

---

## 6. Rủi Ro & Khuyến Nghị Submission

| Tình huống private test      | Model nên dùng              |
| ---------------------------- | --------------------------- |
| Query clean, tương tự public | Profile Hybrid (hiện tại)   |
| Query noisy, format lạ       | Robust Sparse hoặc Router   |
| Không rõ                     | RRF Ensemble (an toàn nhất) |

> **Khuyến nghị:** Đừng submit Profile Hybrid làm final model nếu không có thêm bằng chứng về phân phối private queries. RRF Ensemble sau Phase 3 sẽ là lựa chọn cân bằng tốt nhất giữa precision và recall.

---

## 7. Checklist Ngắn Hạn

- [ ] Implement adaptive threshold, chạy lại eval 13 queries
- [ ] Fix PRF — thêm discriminativeness filter cho expansion terms
- [ ] Thêm WordNet expansion vào Domain Expansion BM25
- [ ] Implement RRF trên 4 model hiện có
- [ ] Debug query 305 và 307 — xem BM25 raw score distribution
- [ ] Implement noise detector cho router
- [ ] Calibrate router threshold trên 13-query eval set
