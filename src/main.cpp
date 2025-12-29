// ------------------- Fix possible "#define new" conflicts -------------------
#pragma push_macro("new")
#ifdef new
#undef new
#endif

#include <new>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>   // FaceDetectorYN

#pragma pop_macro("new")

using namespace std;
namespace fs = std::filesystem;

// ------------------- Utils -------------------
bool isVideoFile(const fs::path& p) {
    string ext = p.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv");
}

bool isImageFile(const fs::path& p) {
    string ext = p.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

string getArg(int argc, char** argv, const string& key, const string& def = "") {
    for (int i = 1; i < argc - 1; i++) {
        if (string(argv[i]) == key) return string(argv[i + 1]);
    }
    return def;
}

bool hasArg(int argc, char** argv, const string& key) {
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == key) return true;
    }
    return false;
}

float cosineSim(const cv::Mat& a, const cv::Mat& b) {
    return a.dot(b); // assumes both are L2 normalized
}

cv::Rect clampRect(const cv::Rect& r, int w, int h) {
    cv::Rect rr = r & cv::Rect(0, 0, w, h);
    if (rr.width <= 0 || rr.height <= 0) return cv::Rect();
    return rr;
}

// ------------------- YuNet Detector (FaceDetectorYN) -------------------
class YuNetDetectorYN {
public:
    struct Face {
        cv::Rect box;
        float score;
    };

    YuNetDetectorYN(const string& modelPath,
                    int inputW = 320, int inputH = 320,
                    float confTh = 0.6f,
                    float nmsTh = 0.4f,
                    int topK = 200)
        : inputW(inputW), inputH(inputH),
          confThreshold(confTh), nmsThreshold(nmsTh), topK(topK)
    {
        detector = cv::FaceDetectorYN::create(
            modelPath,
            "",
            cv::Size(inputW, inputH),
            confThreshold,
            nmsThreshold,
            topK
        );

        if (detector.empty())
            throw runtime_error("Cannot create FaceDetectorYN with model: " + modelPath);
    }

    vector<Face> detect(const cv::Mat& frameBGR, int minFace = 0, int maxFaces = 200) {
        if (frameBGR.empty()) return {};

        detector->setInputSize(frameBGR.size());

        cv::Mat faces;
        detector->detect(frameBGR, faces);

        if (faces.empty()) return {};

        vector<Face> result;
        result.reserve(faces.rows);

        for (int i = 0; i < faces.rows; i++) {
            float x = faces.at<float>(i, 0);
            float y = faces.at<float>(i, 1);
            float w = faces.at<float>(i, 2);
            float h = faces.at<float>(i, 3);
            float score = faces.at<float>(i, 4);

            cv::Rect box((int)x, (int)y, (int)w, (int)h);
            box = clampRect(box, frameBGR.cols, frameBGR.rows);
            if (box.area() <= 0) continue;

            if (minFace > 0 && (box.width < minFace || box.height < minFace))
                continue;

            result.push_back({box, score});
        }

        sort(result.begin(), result.end(), [](const Face& a, const Face& b){
            return a.score > b.score;
        });

        if ((int)result.size() > maxFaces)
            result.resize(maxFaces);

        return result;
    }

private:
    cv::Ptr<cv::FaceDetectorYN> detector;
    int inputW, inputH;
    float confThreshold, nmsThreshold;
    int topK;
};

// ------------------- SFace Recognizer Wrapper -------------------
class SFaceRecognizer {
public:
    SFaceRecognizer(const string& modelPath) {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty())
            throw runtime_error("Cannot load SFace model: " + modelPath);
    }

    cv::Mat getEmbedding(const cv::Mat& faceBGR) {
        cv::Mat resized;
        cv::resize(faceBGR, resized, cv::Size(112, 112));

        cv::Mat blob = cv::dnn::blobFromImage(
            resized, 1.0/128.0, cv::Size(112,112),
            cv::Scalar(127.5,127.5,127.5), true, false);

        net.setInput(blob);
        cv::Mat emb = net.forward(); // 1x512
        cv::normalize(emb, emb, 1.0, 0.0, cv::NORM_L2);
        return emb;
    }

private:
    cv::dnn::Net net;
};

// ------------------- Person DB (CENTROIDS) -------------------
struct PersonDB {
    vector<string> labels;
    vector<cv::Mat> centroids; // 1x512 each
};

static cv::Mat meanEmbedding(const vector<cv::Mat>& embs) {
    cv::Mat mean = cv::Mat::zeros(embs[0].size(), embs[0].type());
    for (auto& e : embs) mean += e;
    mean /= (float)embs.size();
    cv::normalize(mean, mean, 1.0, 0.0, cv::NORM_L2);
    return mean;
}

// ✅ КЛЮЧЕВО: trainAuto читает aligned/ -> clean/ -> root
PersonDB buildDatabaseCentroids(SFaceRecognizer& rec,
                                const string& trainWeb,
                                const string& trainAuto,
                                int maxPerId = 60)
{
    map<string, vector<cv::Mat>> pool;

    auto addFolder = [&](const string& folderPath, const string& label) {
        vector<fs::path> imgs;
        for (auto& e : fs::directory_iterator(folderPath)) {
            if (!e.is_regular_file()) continue;
            if (!isImageFile(e.path())) continue;
            imgs.push_back(e.path());
        }

        sort(imgs.begin(), imgs.end());
        if ((int)imgs.size() > maxPerId) imgs.resize(maxPerId);

        for (auto& p : imgs) {
            cv::Mat img = cv::imread(p.string());
            if (img.empty()) continue;
            pool[label].push_back(rec.getEmbedding(img).clone());
        }
    };

    // ---- Train Web ----
    if (!trainWeb.empty() && fs::exists(trainWeb)) {
        for (auto& e : fs::directory_iterator(trainWeb)) {
            if (!e.is_directory()) continue;
            addFolder(e.path().string(), e.path().filename().string());
        }
    }

    // ---- Train Auto (clusters_clean) ----
    if (!trainAuto.empty() && fs::exists(trainAuto)) {
        for (auto& e : fs::directory_iterator(trainAuto)) {
            if (!e.is_directory()) continue;

            fs::path clusterDir = e.path();
            fs::path alignedDir = clusterDir / "aligned";
            fs::path cleanDir   = clusterDir / "clean";
            string label = clusterDir.filename().string();

            if (fs::exists(alignedDir) && fs::is_directory(alignedDir)) {
                cout << "[DB] " << label << " -> use aligned/\n";
                addFolder(alignedDir.string(), label);
            } else if (fs::exists(cleanDir) && fs::is_directory(cleanDir)) {
                cout << "[DB] " << label << " -> aligned missing, fallback clean/\n";
                addFolder(cleanDir.string(), label);
            } else {
                cout << "[DB] " << label << " -> aligned+clean missing, fallback root\n";
                addFolder(clusterDir.string(), label);
            }
        }
    }

    PersonDB db;
    for (auto& [label, vec] : pool) {
        if (vec.size() < 3) continue; // skip tiny clusters
        cv::Mat c = meanEmbedding(vec);
        db.labels.push_back(label);
        db.centroids.push_back(c);
        cout << "[Train] " << label << " -> " << vec.size() << " images (centroid)\n";
    }

    cout << "[Train] Total identities: " << db.labels.size() << "\n";
    return db;
}

// ------------------- Hungarian (min cost assignment), square matrix -------------------
static vector<int> hungarianMinCost(const vector<vector<double>>& a) {
    int n = (int)a.size();
    const double INF = std::numeric_limits<double>::infinity();

    vector<double> u(n+1), v(n+1);
    vector<int> p(n+1), way(n+1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(n+1, INF);
        vector<char> used(n+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = a[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    vector<int> ans(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0) ans[p[j]-1] = j-1;
    }
    return ans; // row i -> col ans[i]
}

// ------------------- Draw label with background -------------------
void drawLabel(cv::Mat& frame, const string& text, int x, int y, const cv::Scalar& color)
{
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    y = max(y, sz.height + 5);

    cv::rectangle(frame,
                  cv::Rect(x, y - sz.height - 5, sz.width + 10, sz.height + 10),
                  cv::Scalar(0,0,0), cv::FILLED);

    cv::putText(frame, text, cv::Point(x + 5, y + 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
}

// ------------------- Video processing (Hungarian assignment) -------------------
void processVideo(const string& videoPath, const string& outDir,
                  YuNetDetectorYN& det, SFaceRecognizer& rec, const PersonDB& db,
                  int detectEveryN,
                  float unknownThreshold,
                  float marginThreshold,
                  float strongThreshold,
                  int minFace,
                  int maxFaces,
                  bool saveAll)
{
    string vname = fs::path(videoPath).stem().string();
    string outSub = outDir + "/" + vname;
    fs::create_directories(outSub);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open video: " << videoPath << "\n";
        return;
    }

    int frameCount = 0;
    int faceCount = 0;
    int recCount = 0;
    int unkCount = 0;

    const double BIG = 1e6;
    const double nonBestPenalty = 0.05;     // мягко “прижимаем” не-best назначения
    const double dummyCost = std::max(0.0, std::min(1.0, (double)(1.0f - unknownThreshold))); // UNKNOWN cost

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        if (detectEveryN > 1 && frameCount % detectEveryN != 0)
            continue;

        auto faces = det.detect(frame, minFace, maxFaces);
        if (faces.empty()) continue;

        int M = (int)faces.size();
        int N = (int)db.centroids.size();
        int K = std::max(M, N); // square size

        // 1) embeddings для всех лиц
        vector<cv::Mat> embs(M);
        vector<vector<float>> sims(M, vector<float>(N, -1.0f));
        vector<int> bestIdx(M, -1);
        vector<float> bestSim(M, -1.0f), secondSim(M, -1.0f), margin(M, -1.0f);

        for (int i = 0; i < M; i++) {
            cv::Rect box = clampRect(faces[i].box, frame.cols, frame.rows);
            if (box.area() <= 0) continue;

            cv::Mat crop = frame(box).clone();
            if (crop.empty()) continue;

            embs[i] = rec.getEmbedding(crop);

            // sims[i][j]
            float b1 = -1.0f, b2 = -1.0f;
            int bi = -1;
            for (int j = 0; j < N; j++) {
                float s = cosineSim(embs[i], db.centroids[j]);
                sims[i][j] = s;
                if (s > b1) { b2 = b1; b1 = s; bi = j; }
                else if (s > b2) { b2 = s; }
            }
            bestIdx[i] = bi;
            bestSim[i] = b1;
            secondSim[i] = b2;
            margin[i] = (b1 >= 0 && b2 >= 0) ? (b1 - b2) : -1.0f;
        }

        // 2) cost matrix KxK
        vector<vector<double>> cost(K, vector<double>(K, BIG));

        // реальные строки (лица)
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                if (i >= M) {
                    // фиктивная строка (лишняя, если N>M) — пусть назначается куда угодно без влияния
                    cost[i][j] = 0.0;
                    continue;
                }

                if (j < N) {
                    float s = sims[i][j];
                    if (s < 0) { cost[i][j] = BIG; continue; }

                    // Допуск пары:
                    //  - strong всегда допускаем
                    //  - иначе хотя бы sim >= unk_th (margin проверим после назначения)
                    bool allow = (s >= strongThreshold) || (s >= unknownThreshold);
                    if (!allow) {
                        cost[i][j] = BIG;
                        continue;
                    }

                    double c = 1.0 - (double)s;
                    // слегка штрафуем не-best, чтобы венгерский предпочитал best, но мог отдать best другому лицу
                    if (bestIdx[i] >= 0 && j != bestIdx[i]) c += nonBestPenalty;
                    cost[i][j] = c;
                } else {
                    // dummy = UNKNOWN слот
                    cost[i][j] = dummyCost;
                }
            }
        }

        // 3) Hungarian assignment
        auto assign = hungarianMinCost(cost); // size K, row->col

        bool drewSomething = false;

        // 4) Применяем назначение и финальные пороги
        for (int i = 0; i < M; i++) {
            cv::Rect box = clampRect(faces[i].box, frame.cols, frame.rows);
            if (box.area() <= 0) continue;

            int col = (i < (int)assign.size()) ? assign[i] : -1;

            string label = "UNKNOWN";
            bool isUnknown = true;
            float s = -1.0f;

            if (col >= 0 && col < N) {
                s = sims[i][col];

                bool isStrong = (s >= strongThreshold);

                if (isStrong) {
                    isUnknown = false;
                    label = db.labels[col];
                } else if (s >= unknownThreshold) {
                    // margin имеет смысл только для bestIdx (иначе это “вынужденная” подмена)
                    if (col == bestIdx[i] && margin[i] >= marginThreshold) {
                        isUnknown = false;
                        label = db.labels[col];
                    } else {
                        isUnknown = true;
                        label = "UNKNOWN";
                    }
                } else {
                    isUnknown = true;
                    label = "UNKNOWN";
                }
            } else {
                // dummy или невалид — UNKNOWN
                isUnknown = true;
                label = "UNKNOWN";
            }

            faceCount++;
            if (isUnknown) unkCount++; else recCount++;

            cv::Scalar color = isUnknown ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
            cv::rectangle(frame, box, color, 3);

            char buf[256];
            if (s >= 0) snprintf(buf, sizeof(buf), "%s %.2f (m=%.2f)", label.c_str(), s, margin[i]);
            else        snprintf(buf, sizeof(buf), "%s", label.c_str());
            drawLabel(frame, buf, box.x, box.y - 5, color);

            drewSomething = true;
        }

        if (saveAll || drewSomething) {
            char outname[512];
            snprintf(outname, sizeof(outname), "%s/frame_%06d.jpg", outSub.c_str(), frameCount);
            cv::imwrite(outname, frame);
        }
    }

    cout << "[Done] " << videoPath
         << " frames=" << frameCount
         << " faces=" << faceCount
         << " rec=" << recCount
         << " unk=" << unkCount
         << " unk_th=" << unknownThreshold
         << " margin=" << marginThreshold
         << " strong=" << strongThreshold
         << " min_face=" << minFace
         << " max_faces=" << maxFaces
         << "\n";
}

void processDir(const string& dirPath, const string& outDir,
                YuNetDetectorYN& det, SFaceRecognizer& rec, const PersonDB& db,
                int detectEveryN, float unknownThreshold, float marginThreshold, float strongThreshold,
                int minFace, int maxFaces, bool saveAll)
{
    fs::create_directories(outDir);

    vector<fs::path> videos;
    for (auto& e : fs::recursive_directory_iterator(dirPath)) {
        if (e.is_regular_file() && isVideoFile(e.path()))
            videos.push_back(e.path());
    }

    sort(videos.begin(), videos.end());
    cout << "[Batch] Found " << videos.size() << " videos in " << dirPath << "\n";

    for (auto& vp : videos) {
        cout << "\n=== Processing: " << vp.string() << " ===\n";
        processVideo(vp.string(), outDir, det, rec, db,
                     detectEveryN, unknownThreshold, marginThreshold, strongThreshold,
                     minFace, maxFaces, saveAll);
    }
    cout << "\n[Batch] Done. Output saved in " << outDir << "\n";
}

// ------------------- MAIN -------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage:\n";
        cout << "  ./app --dir <video_dir> --out <output_dir>\n";
        cout << "        [--train_web <dir>] [--train_auto <dir>]\n";
        cout << "        [--step N] [--unk_th 0.55] [--margin 0.06] [--strong 0.75]\n";
        cout << "        [--conf 0.6] [--nms 0.4]\n";
        cout << "        [--topk 200] [--max_faces 200] [--min_face 40]\n";
        cout << "        [--max_train_per_id 60]\n";
        cout << "        [--save_all 0/1]\n";
        cout << "        [--yunet model.onnx] [--sface model.onnx]\n";
        return 0;
    }

    string videoDir   = getArg(argc, argv, "--dir");
    string outDir     = getArg(argc, argv, "--out", "../output_batch");

    // disable train_web with: --train_web ""
    string trainWeb   = getArg(argc, argv, "--train_web", "../data/train");
    string trainAuto  = getArg(argc, argv, "--train_auto", "../data/train_auto_pred/clusters");

    int step          = stoi(getArg(argc, argv, "--step", "5"));
    float unkTh       = stof(getArg(argc, argv, "--unk_th", "0.55"));
    float marginTh    = stof(getArg(argc, argv, "--margin", "0.06"));
    float strongTh    = stof(getArg(argc, argv, "--strong", "0.75"));

    float confTh      = stof(getArg(argc, argv, "--conf", "0.60"));
    float nmsTh       = stof(getArg(argc, argv, "--nms",  "0.40"));
    int topK          = stoi(getArg(argc, argv, "--topk", "200"));

    int maxFaces      = stoi(getArg(argc, argv, "--max_faces", "200"));
    int minFace       = stoi(getArg(argc, argv, "--min_face", "40"));

    int maxTrainPerId = stoi(getArg(argc, argv, "--max_train_per_id", "60"));
    bool saveAll      = (stoi(getArg(argc, argv, "--save_all", "0")) != 0);

    string yunetPath  = getArg(argc, argv, "--yunet", "../models/face_detection_yunet_2023mar.onnx");
    string sfacePath  = getArg(argc, argv, "--sface", "../models/face_recognition_sface.onnx");

    if (videoDir.empty()) {
        cerr << "Error: --dir is required\n";
        return -1;
    }

    cout << "=== CONFIG ===\n";
    cout << "dir        : " << videoDir << "\n";
    cout << "out        : " << outDir << "\n";
    cout << "train_web  : " << trainWeb << "\n";
    cout << "train_auto : " << trainAuto << "\n";
    cout << "step       : " << step << "\n";
    cout << "unk_th     : " << unkTh << "\n";
    cout << "margin     : " << marginTh << "\n";
    cout << "strong     : " << strongTh << "\n";
    cout << "conf_th    : " << confTh << "\n";
    cout << "nms_th     : " << nmsTh << "\n";
    cout << "topk       : " << topK << "\n";
    cout << "max_faces  : " << maxFaces << "\n";
    cout << "min_face   : " << minFace << "\n";
    cout << "max_train  : " << maxTrainPerId << "\n";
    cout << "save_all   : " << (saveAll ? "1" : "0") << "\n";
    cout << "yunet      : " << yunetPath << "\n";
    cout << "sface      : " << sfacePath << "\n";
    cout << "===========\n";

    try {
        YuNetDetectorYN det(yunetPath, 320, 320, confTh, nmsTh, topK);
        SFaceRecognizer rec(sfacePath);

        auto db = buildDatabaseCentroids(rec, trainWeb, trainAuto, maxTrainPerId);
        if (db.labels.empty()) {
            cerr << "[FATAL] Empty database (no identities). Check train folders.\n";
            return -1;
        }

        processDir(videoDir, outDir, det, rec, db,
                   step, unkTh, marginTh, strongTh, minFace, maxFaces, saveAll);

    } catch (const exception& e) {
        cerr << "[FATAL] " << e.what() << "\n";
        return -1;
    }

    return 0;
}