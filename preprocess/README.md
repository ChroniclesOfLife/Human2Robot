# Camera RGB-D cố định

Đầu vào: `rgb/00000.png`, `depth/00000.npy` (mét, đã căn chỉnh với RGB),
`intrinsics.json` gồm `k` (3×3), `fps`; `labels.json` ghi true/false cho từng frame
khi tác vụ hoàn tất. Keypoint theo thứ tự 21 điểm chuẩn: wrist=0, thumb-tip=4,
index-tip=8. Frame mất depth/hand/object bị loại và ghi trong manifest.

Chạy với keypoint và mask đã có, không cần tải mô hình:

```bash
python -m preprocess.prepare_rgbd --input /recording \
  --output data/serve_bread/demos/session_002 \
  --hand-backend json --object-backend masks --inpaint none
```

Thêm `hands/00000.json`: `{ "right": { "keypoints": [[u,v], ...] } }`;
`masks/00000/{obj1,obj2,arm}.png`: bánh mì, đĩa, cánh tay. `none` ghi rõ chưa
inpaint.

Chạy các mô hình thị giác với trọng số đã cài:

```bash
python -m pip install -r requirements-preprocess.txt
python -m pip install mediapipe
python -m preprocess.prepare_rgbd --config cfg/preprocess/serve_bread.yaml \
  --input /recording --output data/serve_bread/demos/session_002 \
  --hand-model /weights/hand_landmarker.task \
  --sam2-checkpoint /weights/sam2_hiera_tiny.pt --lama-model /weights/lama_fp32.onnx
```

DINO cần model trong cache hoặc `--dino-model /weights/grounding-dino-tiny`.
`--allow-download` cho phép tải model hỗ trợ. HaMeR: thêm `--hand-backend hamer`
và `--hamer-checkpoint`; cần package HaMeR, cấu hình model và MANO hợp lệ.
Các adapter trọng số ngoài chưa được kiểm chứng trên phiên RGB-D thật ở đây.

Đầu ra: `preprocess/all_data/00000/{training_data.json,observation.png}`.
Tư thế dùng hệ camera cố định, đơn vị mét. Dữ liệu camera mới cần hiệu chuẩn
adapter và huấn luyện phù hợp với quy ước bàn tay/TCP.
