# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

✅ Đã hoàn thành: Triển khai gọi LLM thật (Gemini / OpenAI) và lưu token usage, cost.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

✅ Đã hoàn thành: Triển khai routing policy và guardrail `max_iterations`.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher? -> Khi `sources` rỗng
- Khi nào gọi Analyst? -> Khi có `sources` nhưng chưa có `analysis_notes`
- Khi nào gọi Writer? -> Khi có `analysis_notes` nhưng chưa có `final_answer`
- Khi nào stop? -> Khi có `final_answer` hoặc chạm `max_iterations`
- Nếu agent fail thì retry hay fallback? -> Tenacity retry và fallback Offline Corpus

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

✅ Đã hoàn thành: Triển khai đầy đủ Researcher, Analyst, Writer và Critic agent.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

> **Học viên:** Nguyễn Mai Nhật Anh | **MSSV:** 2A202601826

### 1. Case nào NÊN dùng Multi-Agent? Vì sao?
- **Trường hợp áp dụng**: Các bài toán phức tạp, nhiều công đoạn có tính chất phân hóa chuyên môn rõ rệt như **Nghiên cứu thị trường/kỹ thuật chuyên sâu (Deep Research)**, **Tự động viết code & review (Coder + Reviewer + Tester)**, hoặc **Quy trình thẩm định rủi ro đa tiêu chí**.
- **Lý do dựa trên số liệu thực nghiệm**:
  - **Giảm thiểu ảo giác (Hallucination)**: Tách riêng Researcher để tìm dữ liệu thực tế và Analyst/Critic để kiểm chứng giúp đạt **100% Citation Coverage** (so với 0% của Single-Agent).
  - **Tránh loãng ngữ cảnh (Context Pollution)**: Mỗi agent chỉ tập trung vào một nhiệm vụ với prompt chuyên biệt, cho chất lượng nội dung phân tích sâu hơn vượt trội (**10.0/10 vs 8.0/10**).
  - **Khả năng quan sát & gỡ lỗi (Observability)**: Dễ dàng trace được lỗi phát sinh từ khâu nào (tìm kiếm, lập luận hay hành văn) qua LangSmith/Langfuse.

---

### 2. Case nào KHÔNG NÊN dùng Multi-Agent? Vì sao?
- **Trường hợp áp dụng**: Các tác vụ đơn giản, truy vấn trực tiếp (Direct Q&A), dịch thuật đoạn văn ngắn, tóm tắt một văn bản cho sẵn, hoặc các ứng dụng thời gian thực yêu cầu độ trễ cực thấp (Sub-second latency như chatbot hỗ trợ khách hàng nhanh, voice agent tương tác trực tiếp).
- **Lý do dựa trên số liệu thực nghiệm**:
  - **Độ trễ cao (Latency Overhead)**: Multi-Agent mất nhiều vòng roundtrips qua LLM và điều phối đồ thị, độ trễ tăng từ ~20.9s lên ~37.2s (tăng ~1.8x đến 3x).
  - **Chi phí token cao (Cost Overhead)**: Mỗi agent tiêu tốn thêm token cho prompt hệ thống và ngữ cảnh trao đổi, chi phí tăng gấp ~3.7x ($0.001199 vs $0.000323).
  - **Độ phức tạp hệ thống (System Complexity)**: Việc thiết kế, đồng bộ shared state và xử lý guardrail/vòng lặp vô hạn đòi hỏi chi phí bảo trì hệ thống cao không đáng có đối với các bài toán đơn giản mà Single-Agent với 1 prompt chuẩn đã đủ giải quyết tốt.

