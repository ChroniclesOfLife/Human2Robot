# Kết quả mẫu

`learned_prediction.npz` và `.json`: dự đoán từ checkpoint trên một frame eval,
chạy CPU, seed 7, 20 bước Flow Matching. Đây là dự đoán mô hình; chưa thực thi robot.

`training_smoke.json`: một bước cập nhật trọng số thật và một batch eval riêng.
Loss train 8.0145, eval 7.9418; kiểm tra đường train hoạt động, chưa đánh giá hội tụ.
