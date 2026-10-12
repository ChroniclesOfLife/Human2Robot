# Human2Robot — Serve Bread

Human2Robot chuyển demo con người thành hành động robot:
RGB-D → điểm bàn tay/vật thể → ICT → Flow Matching → tư thế kẹp 6-DoF và đóng/mở kẹp.
Trossen WidowX AI là ví dụ; robot khác cần adapter `RobotArm`, IK và hiệu chuẩn TCP/camera.

Có tác vụ **serve bread**: 868 khung hình train, 864 khung hình eval,
tách theo phiên ghi. Dữ liệu và checkpoint tham chiếu giữ nguyên hình học/trọng số;
xem [nguồn và giấy phép](THIRD_PARTY_NOTICES.md).

## Chạy lại

Python 3.10/3.11; Isaac Sim 6.1 cho mô phỏng. Chạy tại thư mục repo:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_bundle.py --hashes
python -m inference.predict --device cpu
python -m training.FlowMatchingTrainer --task serve_bread --job Human2Robot --use_cfg \
  --device cpu --epochs 1 --batch_size 2 --num_workers 0 --max-steps 1 --eval-batches 1
export ISAAC_SIM_DIR=/duong_dan/isaacsim
H2R_GPU=1 bash scripts/run_isaac.sh --policy-device cpu
```

Train đầy đủ: bỏ các cờ kiểm tra nhanh và dùng `--device cuda:1`.
Máy một GPU: dùng `H2R_GPU=0` và `cuda:0`. Isaac dùng Python đi kèm bản cài đặt;
asset NVIDIA cần mạng lần đầu hoặc cache có sẵn. Kết quả offline:
`outputs/offline_prediction.npz` và `.json`; kết quả Isaac:
`outputs/serve_bread/result.json`.

## Kiểm chứng và tài liệu

Đã kiểm tra dữ liệu/checkpoint, một bước train CPU, suy luận CPU và hai lần
replan Isaac trên GPU 1. Chưa xác nhận chính sách học hoàn tất tác vụ vật lý.
Isaac dùng pose vật thể từ simulator, ẩn robot để tạo ảnh sạch và hỗ trợ kẹp
FixedJoint; `--strict-contact` tắt hỗ trợ. Thành công cần bánh mì được nâng,
thả và nằm ổn định trên đĩa.

[Dữ liệu](data/README.md) · [Xử lý RGB-D mới](preprocess/README.md) ·
[Kết quả mẫu](examples/README.md). Trọng số thị giác tải/cài riêng.
Đưa lên GitHub bằng Git LFS theo `.gitattributes`.
