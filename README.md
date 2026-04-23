# Jetson YOLOv8 USB Cam (TensorRT)

NVIDIA **Jetson AGX Orin** + **JetPack 5.1** 환경에서 USB 카메라 입력을 받아
**YOLOv8**을 **TensorRT (FP16)** 으로 실시간 추론하고, 결과를 화면에 표시하는 C++ 데모입니다.

- TensorRT 8.5 / CUDA 11.4 / OpenCV 4.5 / Ubuntu 20.04
- V4L2 + MJPEG 카메라 캡처
- 단계별(`cap` / `pre` / `trt` / `post` / `draw` / `show`) 타이밍 로깅
- `--nogui` 옵션으로 헤드리스 환경에서 순수 파이프라인 성능 측정

---

## 환경 요구사항

| 항목 | 권장 |
|---|---|
| 보드 | Jetson AGX Orin (다른 Jetson도 동작 가능) |
| OS | Ubuntu 20.04 (JetPack 5.1) |
| TensorRT | 8.5.x |
| CUDA | 11.4 |
| OpenCV | 4.5.x (JetPack 기본 포함) |
| CMake | 3.16+ |
| 카메라 | UVC 호환 USB 카메라 (MJPEG 지원 권장) |

---

## 디렉토리 구조

```
.
├── CMakeLists.txt
├── main.cpp              # 메인 데모 (단계별 타이밍 + --nogui 옵션)
├── main_2.cpp            # 초기 단순 버전 (참고용, 빌드 대상 아님)
├── scripts/
│   └── build_engine.sh   # ONNX → TensorRT (FP16) 엔진 변환 헬퍼
├── LICENSE
└── README.md
```

> `*.onnx`, `*.engine` 파일과 `build/` 디렉토리는 `.gitignore`에 포함되어 있습니다.
> TensorRT 엔진은 빌드한 머신/TensorRT 버전 외에서는 호환되지 않으므로,
> 사용자가 본인 환경에서 직접 빌드하는 것이 안전합니다.

---

## 1. 모델 준비

### 1-1. ONNX 익스포트

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 패키지로 ONNX를 만듭니다.

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True
# → yolov8n.onnx 생성
```

입력 해상도를 바꾸고 싶다면 `imgsz=960` 등을 함께 지정하세요.

```bash
yolo export model=yolov8m.pt format=onnx opset=12 simplify=True imgsz=960
```

### 1-2. TensorRT 엔진 빌드 (FP16)

```bash
chmod +x scripts/build_engine.sh
./scripts/build_engine.sh yolov8n.onnx yolov8n_fp16.engine
```

내부적으로는 JetPack에 포함된 `trtexec`를 사용합니다.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8n.onnx \
    --saveEngine=yolov8n_fp16.engine \
    --fp16
```

---

## 2. 빌드

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 3. 실행

```bash
# 기본: yolov8n_fp16.engine, /dev/video0, 1280x720
./yolo_demo ../yolov8n_fp16.engine 0

# 다른 모델 / 해상도 지정
./yolo_demo ../yolov8m_960_fp16.engine 0
./yolo_demo ../yolov8n_fp16.engine 0 1920 1080

# imshow 끄고 순수 파이프라인 성능만 측정
./yolo_demo ../yolov8n_fp16.engine 0 1280 720 --nogui
```

종료: `ESC` 또는 `q` 키 (`--nogui` 모드는 `Ctrl+C`)

### 인자

| 위치 | 기본값 | 설명 |
|---|---|---|
| 1 | `yolov8n_fp16.engine` | TensorRT 엔진 경로 |
| 2 | `0` | `/dev/videoN` 의 N |
| 3 | `1280` | 캡처 가로 해상도 |
| 4 | `720` | 캡처 세로 해상도 |
| `--nogui` | off | 화면 출력 끔 (헤드리스 측정용) |

---

## 4. 출력 예시

30프레임마다 단계별 평균 시간을 출력합니다.

```
=== Per-stage timing every 30 frames (ms) ===
     cap  |  pre  |  trt  |  post |  draw |  show | total  | fps
----------+-------+-------+-------+-------+-------+--------+------
   3.2   |  4.1  |  6.8  |  1.5  |  0.4  |  2.0  |  18.0  | 55.0
```

| 컬럼 | 의미 |
|---|---|
| `cap`   | `cv::VideoCapture::read()` (V4L2 grab + MJPEG decode) |
| `pre`   | letterbox 리사이즈 + `blobFromImage` |
| `trt`   | H2D memcpy + `enqueueV2` + D2H memcpy + stream sync |
| `post`  | confidence 필터 + NMS + 좌표 역변환 |
| `draw`  | 박스/라벨 그리기 |
| `show`  | `imshow` + `waitKey(1)` |
| `total` | 한 프레임 전체 소요 시간 |

---

## 5. 트러블슈팅

**`Cannot open /dev/videoN`**
- `ls /dev/video*` 로 인덱스를 확인하세요.
- 사용자가 `video` 그룹에 속해 있어야 합니다: `sudo usermod -aG video $USER` (재로그인 필요)

**FPS가 너무 낮음**
- USB 카메라가 MJPEG을 지원하는지 확인:
  `v4l2-ctl --device=/dev/video0 --list-formats-ext`
  (YUYV만 지원하면 720p@30fps가 안 나오는 경우가 많음)
- Jetson 클럭 고정:
  `sudo nvpmodel -m 0 && sudo jetson_clocks`
- `--nogui` 로 imshow 비용을 분리해 측정해 보세요.

**`deserializeCudaEngine failed`**
- 엔진을 빌드했던 머신/TensorRT 버전과 다른 환경에서 실행한 경우입니다.
  현재 환경에서 다시 `scripts/build_engine.sh` 로 엔진을 만드세요.

**xfce4 원격 데스크톱에서 imshow가 깜빡임**
- `cv::namedWindow(..., cv::WINDOW_AUTOSIZE)` 로 고정 크기 사용 중입니다.
  계속 문제가 있으면 `--nogui` 모드를 사용하세요.

---

## 라이선스

[MIT](LICENSE)
