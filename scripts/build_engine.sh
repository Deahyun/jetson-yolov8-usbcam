#!/usr/bin/env bash
# ONNX -> TensorRT (FP16) 엔진 변환 헬퍼
#
# 사용법:
#   ./scripts/build_engine.sh <input.onnx> <output.engine> [추가 trtexec 옵션...]
#
# 예시:
#   ./scripts/build_engine.sh yolov8n.onnx yolov8n_fp16.engine
#   ./scripts/build_engine.sh yolov8m.onnx yolov8m_960_fp16.engine \
#       --shapes=images:1x3x960x960
#
# 환경변수:
#   TRTEXEC  trtexec 실행 파일 경로 (기본: /usr/src/tensorrt/bin/trtexec)

set -euo pipefail

ONNX="${1:-yolov8n.onnx}"
ENGINE="${2:-yolov8n_fp16.engine}"
shift 2 2>/dev/null || true

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ ! -x "$TRTEXEC" ]; then
    echo "Error: trtexec not found at $TRTEXEC" >&2
    echo "JetPack의 TensorRT가 설치되어 있는지 확인하거나 TRTEXEC 환경변수로 경로를 지정하세요." >&2
    exit 1
fi

if [ ! -f "$ONNX" ]; then
    echo "Error: ONNX 파일을 찾을 수 없습니다: $ONNX" >&2
    exit 1
fi

echo "[trtexec] $ONNX -> $ENGINE  (FP16)"
"$TRTEXEC" \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    "$@"

echo "Done: $ENGINE"
