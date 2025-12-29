// ------------------- Fix possible "#define new" conflicts -------------------
#pragma push_macro("new")
#ifdef new
#undef new
#endif

#include <new>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>   // FaceDetectorYN

#pragma pop_macro("new")

using namespace std;
namespace fs = std::filesystem;

// ------------------- Utils -------------------
bool isImageFile(const fs::path& p) {
    string ext = p.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp");
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
    return a.dot(b); // assumes both L2 normalized
}

static cv::Mat meanEmbedding(const vector<cv::Mat>& embs) {
    cv::Mat mean = cv::Mat::zeros(embs[0].size(), embs[0].type());
    for (auto& e : embs) mean += e;
    mean /= (float)embs.size();
    cv::normalize(mean, mean, 1.0, 0.0, cv::NORM_L2);
    return mean;
}

// ------------------- YuNet DetectorYN Wrapper (with landmarks) -------------------
class YuNetDetectorYN {
public:
    YuNetDetectorYN(const string& modelPath,
                    int inputW = 320, int inputH = 320,
                    float confTh = 0.6f,
                    float nmsTh = 0.4f,
                    int topK = 500)
        : inputW(inputW), inputH(inputH),
          confThreshold(confTh), nmsThreshold(nmsTh), topK(topK)
    {
        detector = cv::FaceDetectorYN::create(
            modelPath, "",
            cv::Size(inputW, inputH),
            confThreshold,
            nmsThreshold,
            topK
        );

        if (detector.empty())
            throw runtime_error("Cannot create FaceDetectorYN with model: " + modelPath);
    }

    // return best face row (15 floats) or empty mat
    cv::Mat detectBest(const cv::Mat& frameBGR) {
        if (frameBGR.empty()) return cv::Mat();

        detector->setInputSize(frameBGR.size());

        cv::Mat faces;
        detector->detect(frameBGR, faces);

        // faces: Nx15 (x,y,w,h,score + 10 landmarks)
        if (faces.empty() || faces.rows <= 0) return cv::Mat();

        // pick best score
        int bestIdx = -1;
        float bestScore = -1.0f;
        for (int i = 0; i < faces.rows; i++) {
            float s = faces.at<float>(i, 4);
            if (s > bestScore) {
                bestScore = s;
                bestIdx = i;
            }
        }
        if (bestIdx < 0) return cv::Mat();

        return faces.row(bestIdx).clone();
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

    cv::Mat getEmbeddingAligned112(const cv::Mat& aligned112) {
        // aligned112 expected BGR 112x112
        cv::Mat blob = cv::dnn::blobFromImage(
            aligned112, 1.0/128.0, cv::Size(112,112),
            cv::Scalar(127.5,127.5,127.5), true, false);

        net.setInput(blob);
        cv::Mat emb = net.forward(); // 1x512
        cv::normalize(emb, emb, 1.0, 0.0, cv::NORM_L2);
        return emb;
    }

private:
    cv::dnn::Net net;
};

// ------------------- Alignment using YuNet landmarks -------------------
// YuNet landmarks order in OpenCV FaceDetectorYN output:
// [x, y, w, h, score,  x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm]
// (right eye, left eye, nose, right mouth, left mouth)

static bool alignFace112(const cv::Mat& imgBGR, const cv::Mat& faceRow15, cv::Mat& alignedOut) {
    if (imgBGR.empty() || faceRow15.empty() || faceRow15.cols < 15) return false;

    // Extract 5 keypoints
    cv::Point2f re(faceRow15.at<float>(0,5),  faceRow15.at<float>(0,6));
    cv::Point2f le(faceRow15.at<float>(0,7),  faceRow15.at<float>(0,8));
    cv::Point2f n (faceRow15.at<float>(0,9),  faceRow15.at<float>(0,10));
    cv::Point2f rm(faceRow15.at<float>(0,11), faceRow15.at<float>(0,12));
    cv::Point2f lm(faceRow15.at<float>(0,13), faceRow15.at<float>(0,14));

    vector<cv::Point2f> src = { le, re, n, lm, rm };

    // Standard ArcFace-ish destination points for 112x112
    // (these are common canonical points, good enough for SFace)
    vector<cv::Point2f> dst = {
        {38.2946f, 51.6963f},  // left eye
        {73.5318f, 51.5014f},  // right eye
        {56.0252f, 71.7366f},  // nose
        {41.5493f, 92.3655f},  // left mouth
        {70.7299f, 92.2041f}   // right mouth
    };

    cv::Mat M = cv::estimateAffinePartial2D(src, dst);
    if (M.empty()) return false;

    cv::warpAffine(imgBGR, alignedOut, M, cv::Size(112,112),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    return true;
}

// ------------------- Main cleaner logic -------------------
struct Sample {
    fs::path srcPath;
    fs::path alignedPath;
    cv::Mat emb;      // 1x512
    float simToCentroid = -1.0f;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage:\n";
        cout << "  ./cluster_cleaner --in <clusters_dir> --out <clusters_clean_dir>\n";
        cout << "                  [--yunet model.onnx] [--sface model.onnx]\n";
        cout << "                  [--conf 0.6] [--nms 0.4] [--topk 500]\n";
        cout << "                  [--keep_sim 0.55] [--min_faces 3]\n";
        cout << "                  [--max_per_cluster 200]\n";
        cout << "Example:\n";
        cout << "  ./cluster_cleaner --in ../data/train_auto_pred/clusters --out ../data/train_auto_pred/clusters_clean \\\n";
        cout << "      --yunet ../models/face_detection_yunet_2023mar.onnx --sface ../models/face_recognition_sface.onnx \\\n";
        cout << "      --keep_sim 0.55\n";
        return 0;
    }

    string inDir   = getArg(argc, argv, "--in");
    string outDir  = getArg(argc, argv, "--out");

    string yunetPath = getArg(argc, argv, "--yunet", "../models/face_detection_yunet_2023mar.onnx");
    string sfacePath = getArg(argc, argv, "--sface", "../models/face_recognition_sface.onnx");

    float confTh = stof(getArg(argc, argv, "--conf", "0.60"));
    float nmsTh  = stof(getArg(argc, argv, "--nms",  "0.40"));
    int topK     = stoi(getArg(argc, argv, "--topk", "500"));

    float keepSim = stof(getArg(argc, argv, "--keep_sim", "0.55"));
    int minFaces  = stoi(getArg(argc, argv, "--min_faces", "3"));
    int maxPerCluster = stoi(getArg(argc, argv, "--max_per_cluster", "200"));

    if (inDir.empty() || outDir.empty()) {
        cerr << "[FATAL] --in and --out are required.\n";
        return -1;
    }

    cout << "=== CLUSTER CLEANER CONFIG ===\n";
    cout << "inDir          : " << inDir << "\n";
    cout << "outDir         : " << outDir << "\n";
    cout << "yunet          : " << yunetPath << "\n";
    cout << "sface          : " << sfacePath << "\n";
    cout << "conf_th        : " << confTh << "\n";
    cout << "nms_th         : " << nmsTh << "\n";
    cout << "topk           : " << topK << "\n";
    cout << "keep_sim       : " << keepSim << "\n";
    cout << "min_faces      : " << minFaces << "\n";
    cout << "max_per_cluster: " << maxPerCluster << "\n";
    cout << "==============================\n";

    try {
        YuNetDetectorYN det(yunetPath, 320, 320, confTh, nmsTh, topK);
        SFaceRecognizer rec(sfacePath);

        if (!fs::exists(inDir)) {
            cerr << "[FATAL] Input dir not found: " << inDir << "\n";
            return -1;
        }

        fs::create_directories(outDir);

        int clustersTotal = 0;
        int clustersKept  = 0;
        int totalImgs = 0, totalAligned = 0, totalKept = 0, totalOut = 0;

        for (auto& clusterEntry : fs::directory_iterator(inDir)) {
            if (!clusterEntry.is_directory()) continue;

            clustersTotal++;
            string cname = clusterEntry.path().filename().string();

            vector<fs::path> imgs;
            for (auto& e : fs::directory_iterator(clusterEntry.path())) {
                if (e.is_regular_file() && isImageFile(e.path()))
                    imgs.push_back(e.path());
            }
            sort(imgs.begin(), imgs.end());

            if (imgs.empty()) {
                cout << "[Cluster] " << cname << " -> no images, skip\n";
                continue;
            }

            if ((int)imgs.size() > maxPerCluster)
                imgs.resize(maxPerCluster);

            totalImgs += (int)imgs.size();

            // Output folders
            fs::path outCluster = fs::path(outDir) / cname;
            fs::path outAligned = outCluster / "aligned";
            fs::path outClean   = outCluster / "clean";
            fs::path outOutlier = outCluster / "outliers";

            fs::create_directories(outAligned);
            fs::create_directories(outClean);
            fs::create_directories(outOutlier);

            vector<Sample> samples;
            samples.reserve(imgs.size());

            // 1) Align + embed
            for (auto& imgPath : imgs) {
                cv::Mat img = cv::imread(imgPath.string());
                if (img.empty()) continue;

                cv::Mat bestFace = det.detectBest(img);
                if (bestFace.empty()) continue;

                cv::Mat aligned112;
                if (!alignFace112(img, bestFace, aligned112))
                    continue;

                // save aligned
                fs::path alignedPath = outAligned / imgPath.filename();
                cv::imwrite(alignedPath.string(), aligned112);

                cv::Mat emb = rec.getEmbeddingAligned112(aligned112);

                Sample s;
                s.srcPath = imgPath;
                s.alignedPath = alignedPath;
                s.emb = emb.clone();
                samples.push_back(s);

                totalAligned++;
            }

            if ((int)samples.size() < minFaces) {
                cout << "[Cluster] " << cname << " -> aligned=" << samples.size()
                     << " (< min_faces=" << minFaces << "), skip cluster\n";
                continue;
            }

            // 2) centroid
            vector<cv::Mat> embs;
            embs.reserve(samples.size());
            for (auto& s : samples) embs.push_back(s.emb);

            cv::Mat centroid = meanEmbedding(embs);

            // 3) similarity, split into clean/outliers
            int kept = 0, outl = 0;
            for (auto& s : samples) {
                s.simToCentroid = cosineSim(s.emb, centroid);
            }

            // keep those >= keepSim
            for (auto& s : samples) {
                fs::path dst = (s.simToCentroid >= keepSim) ? (outClean / s.srcPath.filename())
                                                           : (outOutlier / s.srcPath.filename());

                // copy ORIGINAL image (not aligned) into clean/outliers
                try {
                    fs::copy_file(s.srcPath, dst, fs::copy_options::overwrite_existing);
                } catch (...) {
                    // ignore
                }

                if (s.simToCentroid >= keepSim) kept++;
                else outl++;
            }

            totalKept += kept;
            totalOut  += outl;

            clustersKept++;

            cout << "[Cluster] " << cname
                 << " imgs=" << imgs.size()
                 << " aligned=" << samples.size()
                 << " kept=" << kept
                 << " outliers=" << outl
                 << " keep_sim=" << keepSim
                 << "\n";

            // write small report txt
            {
                fs::path rep = outCluster / "report.txt";
                ofstream f(rep.string());
                f << "cluster: " << cname << "\n";
                f << "input_imgs: " << imgs.size() << "\n";
                f << "aligned: " << samples.size() << "\n";
                f << "kept: " << kept << "\n";
                f << "outliers: " << outl << "\n";
                f << "keep_sim: " << keepSim << "\n";
                f << "\nTop 10 lowest sims:\n";
                sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b){
                    return a.simToCentroid < b.simToCentroid;
                });
                for (int i = 0; i < (int)samples.size() && i < 10; i++) {
                    f << samples[i].srcPath.filename().string()
                      << " sim=" << samples[i].simToCentroid << "\n";
                }
            }
        }

        cout << "\n=== SUMMARY ===\n";
        cout << "clusters total     : " << clustersTotal << "\n";
        cout << "clusters processed : " << clustersKept << "\n";
        cout << "images input       : " << totalImgs << "\n";
        cout << "images aligned     : " << totalAligned << "\n";
        cout << "images kept        : " << totalKept << "\n";
        cout << "images outliers    : " << totalOut << "\n";
        cout << "output dir         : " << outDir << "\n";
        cout << "================\n";

    } catch (const exception& e) {
        cerr << "[FATAL] " << e.what() << "\n";
        return -1;
    }

    return 0;
}