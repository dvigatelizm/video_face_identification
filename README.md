# video_face_identification

Face detection + face recognition on video using **OpenCV 4.9**, **YuNet** (detector) and **SFace** (recognizer).

Проект умеет:
- находить лица на видео (YuNet)
- извлекать embeddings (SFace)
- сравнивать с базой известных/кластерных людей
- помечать лица как **KNOWN** (зелёный) или **UNKNOWN** (красный)
- сохранять кадры с bounding boxes и метками

---

## 0) Что внутри

- `src/main.cpp` — основной код распознавания
- `tools/cluster_cleaner.cpp` — очистка кластеров (align + outlier drop)
- `bash_commands/` — удобные команды сборки/запуска
   bash bash_commands/start_recognize_known.bash
   bash bash_commands/start_recognize_unknown.bash
- `models/` — модели YuNet + SFace (скачиваются отдельно)
- `data/` — примеры train/test структуры

---

## 1) Требования (все ОС)

✅ C++17  
✅ CMake >= 3.16  
✅ OpenCV **4.9.0** (или близкий)  
✅ Компилятор: GCC/Clang/MSVC

---

## 2) Скачивание проекта

```bash
git clone https://github.com/dvigatelizm/video_face_identification.git
cd video_face_identification

## 3) Модели (обязательно)
###   Файлы должны лежать в:
models/
  face_detection_yunet_2023mar.onnx
  face_recognition_sface.onnx

##Где взять:
###YuNet:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

###SFace:
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

После скачивания положить их в папку models/.

##4) Установка OpenCV 4.9

###  Linux (Ubuntu/Debian)
###  A: OpenCV уже установлен
pkg-config --modversion opencv4

###  Вариант B: собрать OpenCV 4.9.0 вручную
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev

cd ~
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.9.0
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
sudo make install
sudo ldconfig

### macOS (Homebrew)
### Поставить Xcode Command Line Tools:
xcode-select --install

### Установить OpenCV:
brew update
brew install opencv cmake pkg-config

### Windows (MSVC + vcpkg)
### Установить OpenCV через vcpkg:
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg.exe install opencv4[contrib]:x64-windows

### Сборка проекта через CMake (пример):
cd video_face_identification
mkdir build
cd build

cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -A x64
cmake --build . --config Release

## 5) Сборка проекта (Linux/macOS)
###   Из корня репозитория:
bash bash_commands/build_video_face_identification.bash
build/app

## 6) Очистка кластеров (align + outlier drop)
###   Если хочешь улучшить авто-кластера (clusters/) перед распознаванием:
###6.1 Собрать cleaner:
bash bash_commands/build_claster_cleaner.bash

###6.2 Запустить cleaner:
bash bash_commands/start_claster_clener.bash

###    Результат будет здесь:
data/train_auto_pred/clusters_clean/<cluster>/aligned/*.jpg

## 7) Запуск распознавания (Known / Unknown)
###   Все команды запускаются из корня репозитория.
###7.1 Known (видео где должны быть знакомые люди)
bash bash_commands/start_recognize_known.bash
###   Берёт видео из:
data/test_videos/known/

###7.2 Unknown (видео где людей нет / неизвестные)
bash bash_commands/start_recognize_unknown.bash
###   Берёт видео из:
data/test_videos/unknown/

## 8) Результаты
###   Кадры сохраняются сюда:
output_batch/tmp/<video_name>/*.jpg
🟩 зелёный прямоугольник — распознано как известный человек/кластер
🟥 красный прямоугольник — UNKNOWN

## 9) Параметры распознавания
| параметр      | влияет на        | смысл                                        |
| ------------- | ---------------- | -------------------------------------------- |
| `--step N`    | скорость         | обрабатывать каждый N-й кадр                 |
| `--conf`      | детектор         | порог уверенности детекции YuNet             |
| `--nms`       | детектор         | подавление пересечений прямоугольников       |
| `--topk`      | детектор         | максимум лиц до NMS                          |
| `--min_face`  | детектор         | минимальный размер лица (px)                 |
| `--max_faces` | лимит            | максимум лиц на кадр                         |
| `--unk_th`    | неизвестные      | если bestSim < unk_th => UNKNOWN             |
| `--margin`    | защита от ошибок | если bestSim - secondSim < margin => UNKNOWN |
| `--strong`    | “липкий” режим   | если bestSim >= strong => KNOWN без margin   |

## 10) Пример ручного запуска
cd build

./app --dir ../data/test_videos/known \
      --out ../output_batch/tmp \
      --train_web "" \
      --train_auto ../data/train_auto_pred/clusters_clean \
      --step 5 \
      --conf 0.60 \
      --nms 0.40 \
      --topk 200 \
      --min_face 45 \
      --max_faces 25 \
      --unk_th 0.52 \
      --margin 0.01 \
      --strong 0.75