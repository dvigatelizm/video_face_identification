#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./app <video_path>\n";
        return 0;
    }

    string videoPath = argv[1];

    // 1) Загрузка Haar каскада
    string cascadePath = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cerr << "Error: Cannot load Haar cascade from: " << cascadePath << "\n";
        return -1;
    }

    // 2) Открываем видео
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video: " << videoPath << "\n";
        return -1;
    }

    std::filesystem::create_directories("../output_detected");

    cv::Mat frame;
    int frameCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        // Для ускорения: анализируем только некоторые кадры
        if (frameCount == 1 || frameCount == 50 || frameCount == 70) {

            // Если видео перевернуто, можно повернуть (180 градусов)
            // Закомментируй, если не нужно
            //cv::Mat rotated;
            //cv::rotate(frame, rotated, cv::ROTATE_180);
            cv::Mat rotated = frame;

            // Переводим в grayscale
            cv::Mat gray;
            cv::cvtColor(rotated, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);

            // Детекция лиц
            vector<cv::Rect> faces;
            faceCascade.detectMultiScale(
                gray, faces,
                1.1,        // scaleFactor
                4,          // minNeighbors
                0,
                cv::Size(60, 60) // minSize
            );

            cout << "Frame " << frameCount << " -> faces found: " << faces.size() << "\n";

            // Рисуем bbox
            for (size_t i = 0; i < faces.size(); i++) {
                cv::rectangle(rotated, faces[i], cv::Scalar(0, 255, 0), 2);

                // Можно подписать номер лица
                cv::putText(rotated, "face", cv::Point(faces[i].x, faces[i].y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }

            string outPath = "../output_detected/detected_" + to_string(frameCount) + ".jpg";
            cv::imwrite(outPath, rotated);
            cout << "Saved: " << outPath << "\n";
        }

        if (frameCount > 120) break;
    }

    cout << "Total frames read: " << frameCount << "\n";
    return 0;
}