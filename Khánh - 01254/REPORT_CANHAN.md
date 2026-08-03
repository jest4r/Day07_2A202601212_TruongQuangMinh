# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Ngọc Quốc Khánh
**Nhóm:** TeamB
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Có nghĩa là 2 đối tượng hoặc 2 dữ liệu được biểu diễn bằng vector có hướng rất giống nhau trong không gian vector, đồng nghĩa với việc chúng có mức độ tương đồng cao về bản chất, nội dung hoặc ngữ nghĩa.*


**Ví dụ có độ tương tự CAO:**
- Câu A: Lập trình viên đang sửa lỗi cho ứng dụng
- Câu B: Kỹ sư phần mềm đang xử lý bug trong ứng dụng
- Tại sao tương đồng: dù dùng từ khác nhau nhưng mô hình AI hiểu 2 câu này nói về cùng 1 hành động và đối tượng

**Ví dụ có độ tương tự THẤP:**
- Câu A: Trí tuệ nhân tạo đang phát triển với tốc độ rất nhanh
- Câu B: Hướng dẫn cách sang số khi lái xe ô tô
- Tại sao khác: 1 câu nói về công nghệ 1 câu nói về hành động. 2 câu nằm ở khu vực chủ đề hoàn toàn khác nhau trong không gian vector

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *1. Do không bị ảnh hưởng bởi độ dài văn bản*
> *2. Tránh gặp vấn đề về số chiều*
> *3. Tối ưu hóa tính toán* 
> *4. Mối quan hệ toán học khi vector đã chuẩn hóa*

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính: Bước nhảy = 500 - 50 = 450 ký tự. Số chunk = 1 + ceil((10000 - 500) / 450) = 1 + ceil(9500 / 450) = 1 + 22 = 23.*
> *Đáp án: 23 chunks*

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Số chunk tăng lên 25 (do bước nhảy giảm xuống còn 400). Muốn overlap nhiều hơn để giữ sự liền mạch và tránh làm mất/đứt gãy ngữ cảnh ở ranh giới giữa các chunk.*

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Dùng regex `re.split(r'(?<=[.!?]) |(?<=\.)\n', text.strip())` để tách văn bản theo dấu câu (. ! ?). Xử lý edge case văn bản rỗng trả về `[]`, xóa khoảng trắng thừa từng câu và nhóm tối đa `max_sentences_per_chunk` câu thành 1 chunk.*

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Tách đệ quy theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Base case là khi đoạn văn bản nhỏ hơn `chunk_size` hoặc hết dấu phân cách thì cắt cứng theo `chunk_size`.*

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Lưu trữ dưới dạng danh sách dict gồm `id`, `content`, `metadata`, `embedding`. Khi search thì nhúng câu truy vấn thành vector, tính cosine similarity với từng chunk rồi sắp xếp giảm dần lấy top_k.*

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Lọc trước (pre-filter) các chunk khớp với `metadata_filter` rồi mới tính độ tương tự. Xóa document bằng cách lọc bỏ các chunk có `id` hoặc `metadata['doc_id']` trùng với `doc_id` cần xóa.*

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Tìm top-k chunk liên quan từ store, nối nội dung làm context trong prompt rồi truyền vào LLM để sinh câu trả lời.*

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Lập trình viên đang sửa lỗi cho ứng dụng | Kỹ sư phần mềm đang xử lý bug trong ứng dụng | cao | 0.94 | Đúng |
| 2 | Trí tuệ nhân tạo đang phát triển với tốc độ rất nhanh | Hướng dẫn cách sang số khi lái xe ô tô | thấp | 0.15 | Đúng |
| 3 | Khách hàng có thể đổi trả hàng trong vòng 14 ngày | Thời hạn hoàn trả sản phẩm là 14 ngày kể từ khi nhận | cao | 0.91 | Đúng |
| 4 | Người bán chịu phí vận chuyển nếu giao sai hàng | Thời gian bảo hành sản phẩm là 2 năm toàn EU | thấp | 0.24 | Đúng |
| 5 | Thông tin cá nhân người dùng được bảo mật tuyệt đối | Dữ liệu khách hàng được mã hóa an toàn | cao | 0.89 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Bất ngờ nhất là các câu dùng từ đồng nghĩa khác hẳn mặt chữ nhưng điểm tương đồng vẫn rất cao. Điều này cho thấy embeddings biểu diễn ý nghĩa ngữ nghĩa trong không gian vector chứ không chỉ khớp từ khóa đơn thuần.*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên tập dữ liệu chính sách TMĐT mới (`data/k4_ecommerce/`) bằng chiến lược **`SentenceChunker`** (`max_sentences_per_chunk=3`).
Tổng số lượng chunk nạp vào store: **149 chunks**.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi có bao nhiêu ngày để yêu cầu trả hàng kể từ khi đơn hàng giao thành công? | chinh-sach-van-chuyen: Khiếu nại với đơn trả hàng hoàn tiền... | 0.842 | Không (Top-1) | Trả về chunk thuộc chính sách vận chuyển do trùng từ khóa "trả hàng". |
| 2 | Thời gian xử lý bảo hành dự kiến là bao lâu? | chinh-sach-bao-hanh: THỜI GIAN BẢO HÀNH: a. Bảo hành tại nhà sản xuất... | 0.800 | Có | Trả về đúng chunk quy định thời gian bảo hành sản phẩm. |
| 3 | Đơn hàng nào không hỗ trợ vận chuyển? | chinh-sach-van-chuyen: Quy định về giới hạn đơn hàng và giao hàng không thành công... | 0.821 | Có | Trả về đúng chunk thuộc chính sách vận chuyển. |
| 4 | Lịch sử trò chuyện với chăm sóc khách hàng lưu trữ tối đa bao lâu? | chinh-sach-van-chuyen: Khiếu nại vận chuyển đơn giao không thành công... | 0.779 | Không (Top-1) | Bị nhiễu bởi các đoạn văn bản dài không chứa dấu chấm câu rõ ràng. |
| 5 *(lọc metadata)* | Người bán vi phạm chính sách sẽ bị áp dụng những chế tài nào? | quy-dinh-dang-ban-san-pham: Sản phẩm thuốc không kê đơn và danh mục hạn chế... | 0.828 | Không (Top-1) | Lọc metadata giúp giới hạn vai trò người bán, nhưng SentenceChunker ngắt đứt tiêu đề làm trôi ngữ cảnh. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **2** / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Phân tích thất bại (Failure Analysis): `SentenceChunker` tách văn bản dựa trên dấu chấm (`. `), nhưng tài liệu Markdown chính sách có rất nhiều tiêu đề và danh sách gạch đầu dòng không có dấu chấm. Điều này khiến `SentenceChunker` dính nhiều đoạn văn bản khác chủ đề vào làm một chunk dài, gây nhiễu ngữ cảnh. Chiến lược `RecursiveChunker` hoặc `MarkdownHeadingChunker` vượt trội hơn đối với tập dữ liệu dạng Markdown.*




---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
