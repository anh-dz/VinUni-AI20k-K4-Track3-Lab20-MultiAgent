# Multi-Agent Research System - Architecture & Design Document

## Problem
Xây dựng một hệ thống AI Research Assistant tự động hóa quy trình nghiên cứu chuyên sâu: nhận truy vấn phức tạp từ người dùng, tìm kiếm nguồn tài liệu thực tế, phân tích đánh giá đa chiều, tổng hợp luận điểm và soạn thảo báo cáo kỹ thuật hoàn chỉnh kèm trích dẫn nguồn minh bạch (inline citations & references).

## Why multi-agent?
Một agent đơn lẻ ("single-agent làm tất cả") thường gặp các hạn chế nghiêm trọng khi xử lý bài toán nghiên cứu dài:
1. **Loãng ngữ cảnh (Context Fragmentation & Pollution)**: Nhồi nhét cả prompt tìm kiếm, nội dung tài liệu thô, phân tích logic và định dạng báo cáo vào một prompt khiến mô hình dễ bỏ sót yêu cầu và giảm chất lượng suy luận.
2. **Ảo giác và thiếu kiểm chứng (Hallucination without Grounding)**: Single-agent có xu hướng tự tạo câu trả lời dựa trên trọng số huấn luyện thay vì dựa vào các bài báo/tài liệu nghiên cứu mới nhất.
3. **Khó kiểm soát và gỡ lỗi (Lack of Observability & Modularity)**: Khi xảy ra sai sót, không thể biết lỗi nằm ở bước tìm kiếm, bước phân tích hay bước hành văn.

Multi-Agent phân rã bài toán thành các chuyên trách độc lập (Supervisor, Researcher, Analyst, Writer, Critic) giúp tối ưu hóa prompt cho từng vai trò và kiểm soát chất lượng qua từng chặng.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng làm việc, quyết định agent tiếp theo và ngắt vòng lặp | `ResearchState` | `next_route` ('researcher', 'analyst', 'writer', 'done') | Vòng lặp vô hạn nếu không có guardrail `max_iterations` |
| **Researcher** | Tìm kiếm tài liệu ngoài qua Tavily/Offline corpus và tóm tắt factual findings | `state.request` | `state.sources`, `state.research_notes` | Không tìm thấy nguồn hoặc nguồn rác / irrelevant |
| **Analyst** | Phân tích phản biện, so sánh các quan điểm, đánh giá trade-off và trích xuất insights | `state.research_notes`, `state.sources` | `state.analysis_notes` | Phân tích nông, bỏ sót luận điểm chính hoặc sai lệch ngữ cảnh |
| **Writer** | Soạn thảo báo cáo kỹ thuật có cấu trúc hoàn chỉnh và gắn inline citations `[1]`, `[2]` | `state.analysis_notes`, `state.sources` | `state.final_answer` | Không gắn trích dẫn hoặc format sai định dạng chuẩn |
| **Critic** | Kiểm định tính xác thực, kiểm tra độ phủ trích dẫn và phát hiện ảo giác | `state.final_answer`, `state.sources` | `AgentResult` với `citation_coverage` | Bỏ sót lỗi trích dẫn giả mạo |

## Shared state (`ResearchState`)

- `request`: Chứa truy vấn gốc (`query`), số lượng nguồn tối đa (`max_sources`), và đối tượng độc giả (`audience`).
- `iteration`: Bộ đếm số chu kỳ thực thi để phục vụ guardrail.
- `route_history`: Lịch sử các bước chuyển tiếp agent (`['researcher', 'analyst', 'writer', 'done']`).
- `sources`: Danh sách `SourceDocument` (title, url, snippet, metadata) làm căn cứ trích dẫn.
- `research_notes`: Ghi chú tóm tắt dữ liệu thực tế từ Researcher.
- `analysis_notes`: Báo cáo phân tích chuyên sâu và so sánh từ Analyst.
- `final_answer`: Bài viết báo cáo hoàn chỉnh cuối cùng từ Writer.
- `agent_results`: Danh sách `AgentResult` ghi nhận token, latency và chi phí từng bước.
- `trace`: Mảng sự kiện trace phục vụ observability (LangSmith / Langfuse).
- `errors`: Danh sách lỗi phát sinh để fallback.

## Routing policy

```text
[START]
   │
   ▼
[supervisor] ◄────────────────────────┐
   │                                  │
   ├─► (chưa có sources) ───────────► [researcher] ──┤
   ├─► (chưa có analysis_notes) ────► [analyst] ─────┤
   ├─► (chưa có final_answer) ──────► [writer] ──────┤
   └─► (đã có answer || iter >= max) ─► [done / END]
```

## Guardrails

- **Max iterations**: Giới hạn tối đa 6 chu kỳ điều phối (`MAX_ITERATIONS=6`) để ngắt dứt điểm các vòng lặp vô hạn.
- **Timeout**: Timeout ở mức 60s cho mỗi lần gọi network/API (`TIMEOUT_SECONDS=60`).
- **Retry**: Áp dụng Tenacity Exponential Backoff Retry (3 lần) cho các lỗi tạm thời khi gọi LLM.
- **Fallback**: Tự động fallback sang kho tài liệu Offline Corpus (`ai_agent_offline_research_corpus_v2`) nếu Tavily search gặp sự cố mạng/hết quota.
- **Validation**: Schema validation nghiêm ngặt với Pydantic v2 cho toàn bộ state và truy vấn đầu vào.

## Benchmark plan

- **Dataset / Query**: Kiểm thử trên các câu hỏi nghiên cứu kỹ thuật phức tạp (ví dụ: *"Research GraphRAG state-of-the-art and write a comprehensive summary"*).
- **Metrics**:
  - **Latency (s)**: Thời gian hoàn thành toàn chu trình.
  - **Estimated Cost (USD)**: Tổng chi phí token tiêu thụ.
  - **Quality Score (0-10)**: Điểm chất lượng cấu trúc, chiều sâu và độ tin cậy.
  - **Citation Coverage (%)**: Tỷ lệ các tài liệu tham khảo được trích dẫn chính xác trong bài viết.
  - **Failure Rate (%)**: Tỷ lệ lỗi / phản hồi rỗng.
- **Expected Outcome**: Multi-Agent cho chất lượng vượt trội và citation coverage đạt 100% (so với 0% của Baseline), chấp nhận trade-off latency cao hơn và token tiêu thụ nhiều hơn.
