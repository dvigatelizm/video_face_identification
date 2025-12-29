import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import face_recognition
import csv

TRAIN_AUTO = "/home/r2d2/video_face_identification/data/train_auto"
OUT_DIR = "/home/r2d2/video_face_identification/data/train_auto_pred"
CLUSTERS_DIR = os.path.join(OUT_DIR, "clusters")
NOISE_DIR = os.path.join(OUT_DIR, "noise")

# Параметры кластеризации
EPS_LIST = [0.30, 0.33, 0.35, 0.38, 0.40]          # чем меньше — тем строже (0.35..0.60)
MIN_SAMPLES = 4     # минимальный размер кластера

def collect_images(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(dirpath, f))
    return sorted(files)

def get_embedding(img_path):
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        return None
    # если на фото несколько лиц (бывает), берем первое
    return encodings[0]

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def main():
    safe_mkdir(OUT_DIR)
    safe_mkdir(CLUSTERS_DIR)
    safe_mkdir(NOISE_DIR)

    print("[1] Collecting images...")
    images = collect_images(TRAIN_AUTO)
    print(f"Found {len(images)} images")

    embeddings = []
    valid_images = []

    print("[2] Computing embeddings (this can take time)...")
    for img_path in tqdm(images):
        emb = get_embedding(img_path)
        if emb is None:
            continue
        embeddings.append(emb)
        valid_images.append(img_path)

    embeddings = np.array(embeddings)
    print(f"Embeddings computed for {len(valid_images)} images")

    print("[3] Clustering with DBSCAN...")
    #clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean").fit(embeddings)
    #labels = clustering.labels_
    best = None
    best_labels = None
    best_eps = None
    
    for EPS in EPS_LIST:
        clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean").fit(embeddings)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        print(f"[Try EPS={EPS}] clusters={n_clusters} noise={n_noise} total={len(labels)}")
        
        # выбираем тот EPS, где кластеров много, но noise не запредельный
        if n_clusters > 10 and (best is None or (n_clusters - n_noise*0.01) > best):
            best = (n_clusters - n_noise*0.01)
            best_labels = labels
            best_eps = EPS
    
    if best_labels is None:
        print("Could not find good EPS in EPS_LIST, using last one")
        best_labels = labels
        best_eps = EPS_LIST[-1]
    
    labels = best_labels
    print(f"[Selected EPS={best_eps}]")


    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise samples: {n_noise}")

    # CSV report
    csv_path = os.path.join(OUT_DIR, "clustering.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "cluster_id"])

        print("[4] Copying files into cluster folders...")
        for img_path, cid in tqdm(zip(valid_images, labels), total=len(valid_images)):
            writer.writerow([img_path, cid])

            base = os.path.basename(img_path)
            if cid == -1:
                out_path = os.path.join(NOISE_DIR, base)
            else:
                cluster_path = os.path.join(CLUSTERS_DIR, f"cluster_{cid:04d}")
                safe_mkdir(cluster_path)
                out_path = os.path.join(cluster_path, base)

            # чтобы не копировать одинаковые имена — добавим префикс папки
            parent = os.path.basename(os.path.dirname(img_path))
            out_path = out_path.replace(base, f"{parent}_{base}")
            shutil.copy(img_path, out_path)

    print(f"\nDONE ✅")
    print(f"Clusters saved in: {CLUSTERS_DIR}")
    print(f"Noise saved in: {NOISE_DIR}")
    print(f"CSV report: {csv_path}")

if __name__ == "__main__":
    main()
