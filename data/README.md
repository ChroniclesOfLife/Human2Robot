# Dữ liệu serve bread

`serve_bread/demos/session_001`: 868 khung hình huấn luyện.
`serve_bread/demos/session_000`: 864 khung hình đánh giá; tách theo phiên ghi.

Mỗi frame gồm `observation.png` và `training_data.json`: pose bàn tay, trạng thái
kẹp, pose bánh mì/đĩa và hiệu chuẩn camera. Đơn vị vị trí: mét.
`serve_bread/manifest.json` và manifest từng session ghi số lượng, split và SHA-256.

Đây là demo tham chiếu đã xử lý; xem [nguồn dữ liệu](../THIRD_PARTY_NOTICES.md).
Train trực tiếp bằng dữ liệu này. Dữ liệu camera cố định mới dùng
[hướng dẫn RGB-D](../preprocess/README.md).
