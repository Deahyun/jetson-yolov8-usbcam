// main.cpp
// Jetson AGX Orin + USB Cam + YOLOv8 + TensorRT 데모 (JetPack 5.1 기준)
// - TensorRT 8.5.2 / CUDA 11.4 / OpenCV 4.5.x / Ubuntu 20.04
// - xfce4 원격 데스크톱에서 cv::imshow 표시 가능
//
// 빌드:  mkdir build && cd build && cmake .. && make -j$(nproc)
// 실행:  ./yolo_demo ../yolov8n_fp16.engine 0

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace nvinfer1;

// ---------- TensorRT Logger ----------
class TrtLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};
static TrtLogger gLogger;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

struct Detection {
    cv::Rect box;
    float    conf;
    int      classId;
};

// COCO 80 classes (YOLOv8 기본)
static const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// ---------- YOLOv8 TensorRT 래퍼 (TensorRT 8.x API) ----------
class YoloV8Trt {
public:
    YoloV8Trt(const std::string& enginePath,
              float confThresh = 0.3f,
              float nmsThresh  = 0.45f)
        : m_confThresh(confThresh), m_nmsThresh(nmsThresh) {
        loadEngine(enginePath);
        allocateBuffers();
    }

    ~YoloV8Trt() {
        for (auto* p : m_devBuffers) {
            if (p) cudaFree(p);
        }
        if (m_stream) cudaStreamDestroy(m_stream);
    }

    std::vector<Detection> detect(const cv::Mat& img) {
        cv::Mat blob = preprocess(img);

        // Host → Device
        CUDA_CHECK(cudaMemcpyAsync(
            m_devBuffers[m_inputIdx], blob.ptr<float>(),
            m_inputElems * sizeof(float),
            cudaMemcpyHostToDevice, m_stream));

        // TensorRT 8.x: enqueueV2 (bindings 배열 방식)
        if (!m_context->enqueueV2(m_devBuffers.data(), m_stream, nullptr)) {
            std::cerr << "enqueueV2 failed" << std::endl;
            return {};
        }

        // Device → Host
        std::vector<float> output(m_outputElems);
        CUDA_CHECK(cudaMemcpyAsync(
            output.data(), m_devBuffers[m_outputIdx],
            m_outputElems * sizeof(float),
            cudaMemcpyDeviceToHost, m_stream));
        CUDA_CHECK(cudaStreamSynchronize(m_stream));

        return postprocess(output, img.cols, img.rows);
    }

private:
    std::unique_ptr<IRuntime>          m_runtime;
    std::unique_ptr<ICudaEngine>       m_engine;
    std::unique_ptr<IExecutionContext> m_context;
    std::vector<void*>                 m_devBuffers;
    cudaStream_t                       m_stream = nullptr;

    int    m_inputIdx    = -1;
    int    m_outputIdx   = -1;
    int    m_inputW      = 640;
    int    m_inputH      = 640;
    size_t m_inputElems  = 0;
    size_t m_outputElems = 0;
    int    m_numClasses  = 80;
    int    m_numBoxes    = 8400;

    float m_confThresh;
    float m_nmsThresh;
    float m_scale = 1.0f;
    int   m_padX  = 0;
    int   m_padY  = 0;

    void loadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open engine file: " + path);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        file.read(data.data(), size);

        m_runtime.reset(createInferRuntime(gLogger));
        if (!m_runtime) throw std::runtime_error("createInferRuntime failed");

        m_engine.reset(m_runtime->deserializeCudaEngine(data.data(), size));
        if (!m_engine) throw std::runtime_error("deserializeCudaEngine failed");

        m_context.reset(m_engine->createExecutionContext());
        if (!m_context) throw std::runtime_error("createExecutionContext failed");

        CUDA_CHECK(cudaStreamCreate(&m_stream));
    }

    void allocateBuffers() {
        // TensorRT 8.x API: getNbBindings / bindingIsInput / getBindingDimensions
        int nBindings = m_engine->getNbBindings();
        m_devBuffers.assign(nBindings, nullptr);

        for (int i = 0; i < nBindings; ++i) {
            Dims       dims    = m_engine->getBindingDimensions(i);
            const char* name   = m_engine->getBindingName(i);
            bool       isInput = m_engine->bindingIsInput(i);

            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];

            CUDA_CHECK(cudaMalloc(&m_devBuffers[i], vol * sizeof(float)));

            if (isInput) {
                m_inputIdx   = i;
                m_inputElems = vol;
                if (dims.nbDims == 4) {   // [1,3,H,W]
                    m_inputH = dims.d[2];
                    m_inputW = dims.d[3];
                }
                std::cout << "[IO] input[" << i << "] '" << name << "' shape=["
                          << dims.d[0] << "," << dims.d[1] << ","
                          << dims.d[2] << "," << dims.d[3] << "]\n";
            } else {
                m_outputIdx   = i;
                m_outputElems = vol;
                if (dims.nbDims == 3) {   // YOLOv8: [1, 4+C, N]
                    m_numClasses = dims.d[1] - 4;
                    m_numBoxes   = dims.d[2];
                }
                std::cout << "[IO] output[" << i << "] '" << name << "' shape=["
                          << dims.d[0] << "," << dims.d[1] << ","
                          << dims.d[2] << "] classes=" << m_numClasses
                          << " boxes=" << m_numBoxes << "\n";
            }
        }

        if (m_inputIdx < 0 || m_outputIdx < 0)
            throw std::runtime_error("Cannot find input/output binding");
    }

    cv::Mat preprocess(const cv::Mat& img) {
        // Letterbox 리사이즈 (aspect ratio 유지 + 114 회색 패딩)
        m_scale = std::min(static_cast<float>(m_inputW) / img.cols,
                           static_cast<float>(m_inputH) / img.rows);
        int newW = static_cast<int>(std::round(img.cols * m_scale));
        int newH = static_cast<int>(std::round(img.rows * m_scale));
        m_padX = (m_inputW - newW) / 2;
        m_padY = (m_inputH - newH) / 2;

        cv::Mat resized, padded;
        cv::resize(img, resized, cv::Size(newW, newH));
        cv::copyMakeBorder(resized, padded,
            m_padY, m_inputH - newH - m_padY,
            m_padX, m_inputW - newW - m_padX,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        cv::Mat blob;
        // BGR→RGB (swapRB=true), /255.0, HWC→NCHW float32
        cv::dnn::blobFromImage(padded, blob,
            1.0 / 255.0,
            cv::Size(m_inputW, m_inputH),
            cv::Scalar(), /*swapRB=*/true, /*crop=*/false, CV_32F);
        return blob;
    }

    std::vector<Detection> postprocess(const std::vector<float>& out,
                                       int origW, int origH) {
        // YOLOv8 output layout: [1, 4+C, N]
        //  out[(4+c)*N + i] = class c 의 i번째 앵커 confidence
        //  out[k*N + i]  (k=0~3) = 박스 cx, cy, w, h
        std::vector<cv::Rect> boxes;
        std::vector<float>    confs;
        std::vector<int>      ids;
        boxes.reserve(256); confs.reserve(256); ids.reserve(256);

        const int N = m_numBoxes;
        const int C = m_numClasses;

        for (int i = 0; i < N; ++i) {
            float maxConf = 0.0f;
            int   maxCls  = 0;
            for (int c = 0; c < C; ++c) {
                float v = out[(4 + c) * N + i];
                if (v > maxConf) { maxConf = v; maxCls = c; }
            }
            if (maxConf < m_confThresh) continue;

            float cx = out[0 * N + i];
            float cy = out[1 * N + i];
            float w  = out[2 * N + i];
            float h  = out[3 * N + i];

            // letterbox 역변환 → 원본 좌표계
            float x  = (cx - w * 0.5f - m_padX) / m_scale;
            float y  = (cy - h * 0.5f - m_padY) / m_scale;
            float bw = w / m_scale;
            float bh = h / m_scale;

            boxes.emplace_back(
                static_cast<int>(x),  static_cast<int>(y),
                static_cast<int>(bw), static_cast<int>(bh));
            confs.push_back(maxConf);
            ids.push_back(maxCls);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, confs, m_confThresh, m_nmsThresh, keep);

        std::vector<Detection> result;
        result.reserve(keep.size());
        cv::Rect frameRect(0, 0, origW, origH);
        for (int idx : keep) {
            Detection d;
            d.box     = boxes[idx] & frameRect;   // 화면 밖 클리핑
            d.conf    = confs[idx];
            d.classId = ids[idx];
            if (d.box.width > 0 && d.box.height > 0)
                result.push_back(d);
        }
        return result;
    }
};

// ---------- 색상 테이블 ----------
static cv::Scalar colorFor(int id) {
    static const cv::Scalar palette[] = {
        {255, 56, 56}, {255,157,151}, {255,112, 31}, {255,178, 29}, {207,210, 49},
        { 72,249, 10}, {146,204, 23}, { 61,219,134}, { 26,147, 52}, {  0,212,187},
        { 44,153,168}, {  0,194,255}, { 52, 69,147}, {100,115,255}, {  0, 24,236},
        {132, 56,255}, { 82,  0,133}, {203, 56,255}, {255,149,200}, {255, 55,199}
    };
    return palette[id % (sizeof(palette) / sizeof(palette[0]))];
}

// ---------- main ----------
int main(int argc, char** argv) {
    std::string enginePath = "yolov8n_fp16.engine";
    int camIdx = 0;
    int capW   = 1280;
    int capH   = 720;
    int capFps = 30;

    if (argc > 1) enginePath = argv[1];
    if (argc > 2) camIdx     = std::stoi(argv[2]);
    if (argc > 3) capW       = std::stoi(argv[3]);
    if (argc > 4) capH       = std::stoi(argv[4]);

    std::cout << "Engine : " << enginePath << "\n"
              << "Camera : /dev/video" << camIdx
              << " (" << capW << "x" << capH << " @ " << capFps << "fps)\n";

    // 1) 디텍터 로드
    std::unique_ptr<YoloV8Trt> detector;
    try {
        detector = std::make_unique<YoloV8Trt>(enginePath, 0.3f, 0.45f);
    } catch (const std::exception& e) {
        std::cerr << "Detector init failed: " << e.what() << std::endl;
        return 1;
    }

    // 2) 카메라 오픈 (V4L2 + MJPEG 강제)
    cv::VideoCapture cap(camIdx, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open /dev/video" << camIdx << std::endl;
        return 1;
    }
    // MJPEG 지정 안 하면 YUYV 기본이라 720p@30fps가 안 나오는 캠이 많음
    cap.set(cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  capW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, capH);
    cap.set(cv::CAP_PROP_FPS,          capFps);
    cap.set(cv::CAP_PROP_BUFFERSIZE,   1);  // 지연 최소화

    std::cout << "Actual : "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH)  << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << "fps\n";

    // 3) 창 생성
    const std::string winName = "YOLOv8 USB Cam (ESC or q to quit)";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    auto   tPrev   = std::chrono::steady_clock::now();
    int    frames  = 0;
    double fpsAvg  = 0.0;
    double inferMs = 0.0;

    // 4) 메인 루프
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Grab failed, retrying...\n";
            cv::waitKey(10);
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        auto dets = detector->detect(frame);
        auto t1 = std::chrono::steady_clock::now();
        inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 디텍션 그리기
        for (const auto& d : dets) {
            cv::Scalar color = colorFor(d.classId);
            cv::rectangle(frame, d.box, color, 2);

            std::string label = cv::format("%s %.2f",
                COCO_NAMES[d.classId].c_str(), d.conf);
            int baseLine = 0;
            cv::Size sz = cv::getTextSize(label,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int yTop = std::max(d.box.y, sz.height + 4);
            cv::rectangle(frame,
                cv::Point(d.box.x,              yTop - sz.height - 4),
                cv::Point(d.box.x + sz.width+4, yTop),
                color, cv::FILLED);
            cv::putText(frame, label,
                cv::Point(d.box.x + 2, yTop - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
        }

        // FPS 집계 (0.5초마다 평균)
        auto tNow = std::chrono::steady_clock::now();
        double dtSec = std::chrono::duration<double>(tNow - tPrev).count();
        if (dtSec >= 0.5) {
            fpsAvg = frames / dtSec;
            frames = 0;
            tPrev  = tNow;
        }
        ++frames;

        std::string info = cv::format("FPS: %.1f | Infer: %.1fms | Dets: %zu",
            fpsAvg, inferMs, dets.size());
        cv::putText(frame, info, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        cv::imshow(winName, frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
