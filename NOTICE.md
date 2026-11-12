# Nguồn gốc và ghi công

DataMate Agent — Locked-In1 được Nguyen Han (@hana200804) phát triển và duy trì tại:
https://github.com/hana200804/datamate-agent-Locked-In1

## Mã kế thừa

Bản mã ban đầu được tiếp nhận từ dự án DataMate của hugebenevolence:
https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26

Tài liệu đi kèm bản mã đó ghi nhóm tác giả: Dương Trường Bình, Tô Phát Đạt, Trần Đại Nhân; giấy phép MIT với thông báo `Copyright (c) 2026 The DataMate authors`. Thông báo này được giữ trong LICENSE cho phần mã kế thừa. Các thay đổi và phần bổ sung của dự án này thuộc Nguyen Han.

Ngân hàng few-shot, các split ID và một số tài liệu kỹ thuật/hình minh họa được tiếp nhận cùng mã nguồn. Chúng không phải bằng chứng về việc train hay điểm số của phiên bản hiện tại. Lịch sử điểm v1–v9 được giữ riêng tại [docs/archive/upstream-results.md](docs/archive/upstream-results.md).

Adapter `sql_rft_v3`, view public lưu sẵn và các báo cáo cũ chỉ giữ trong checkout cục bộ nếu đã có; không được xuất bản hoặc tự tải trong luồng chạy hiện tại. Muốn dùng LoRA, chọn adapter từ lượt train của mình.

## Thành phần ngoài

- Qwen3.5-9B: model gốc, tải riêng; thông tin revision và giấy phép trong `docker/model_manifest.json`.
- vLLM, DuckDB, pandas, Unsloth, PEFT và flash-linear-attention: các thư viện được dự án sử dụng, giữ giấy phép riêng.
- Dữ liệu Zalo AI Challenge: do ban tổ chức cung cấp, không nằm trong giấy phép mã nguồn và không được đóng gói vào repo public.
