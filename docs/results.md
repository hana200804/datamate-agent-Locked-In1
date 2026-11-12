# Trạng thái kết quả

## Public base G4

Checkout cục bộ có `runs/public_base_g4/submission.jsonl`; sơ đồ luồng ghi nhận PA **0.680**, 846 câu trả lời và 154 câu bỏ qua, thời gian khoảng 72 phút trên G4. Đây là số được ghi lại trong quá trình làm dự án; lần dọn repo này không chạy lại GPU hoặc xác minh điểm trên portal.

Kiểm tra lại cấu trúc file nộp khi có dữ liệu câu hỏi bằng `python -m datamate.submission check`. Kết quả và log đầy đủ được giữ trong `runs/`, không đưa vào Git.

## Kiểm chứng phiên bản hiện tại

- Kiểm thử CPU bằng `python -m pytest -q`.
- Chạy `scripts/colab_train.py --dry-run` để xem trước luồng Colab.
- LoRA tự train và Docker cần được đánh giá trên lượt chạy riêng trước khi công bố điểm hoặc thời gian.

Khi ghi nhận kết quả mới, lưu cấu hình, commit, model/revision, GPU, thời gian, số câu trả lời và điểm portal hoặc báo cáo chấm train. Lịch sử của mã nguồn gốc được tách tại [archive/upstream-results.md](archive/upstream-results.md).
