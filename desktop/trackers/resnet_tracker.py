from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================
# CONFIG
# ==========================

SIMILARITY_THRESHOLD = 0.80
MEMORY_TIMEOUT = 30

# ==========================
# YOLO
# ==========================

model = YOLO("best.pt")

# ==========================
# RESNET50 FEATURE EXTRACTOR
# ==========================

weights = models.ResNet50_Weights.DEFAULT

resnet = models.resnet50(weights=weights)

resnet.fc = nn.Identity()

resnet.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# MEMORY
# ==========================

known_objects = {}

track_to_global = {}

next_global_id = 1

counts = {
    0: 0,
    1: 0,
    2: 0
}

class_names = {
    0: "Bottle",
    1: "Cup",
    2: "Remote"
}

# ==========================
# FEATURE EXTRACTION
# ==========================

def get_embedding(frame, x1, y1, x2, y2):

    roi = frame[
        max(0, y1):max(y2, y1 + 1),
        max(0, x1):max(x2, x1 + 1)
    ]

    if roi.size == 0:
        return None

    img = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        feat = resnet(img)

    feat = feat.cpu().numpy().flatten()

    feat = feat / np.linalg.norm(feat)

    return feat

# ==========================
# TRACKING
# ==========================

results = model.track(
    source=0,
    tracker="botsort.yaml",
    conf=0.60,
    persist=True,
    stream=True
)

for r in results:

    frame = r.orig_img.copy()

    current_time = time.time()

    expired = []

    for gid, obj in known_objects.items():

        if current_time - obj["last_seen"] > MEMORY_TIMEOUT:

            expired.append(gid)

    for gid in expired:

        del known_objects[gid]

    if r.boxes.id is not None:

        ids = r.boxes.id.cpu().numpy().astype(int)

        classes = r.boxes.cls.cpu().numpy().astype(int)

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

        for track_id, cls, box in zip(ids, classes, boxes):

            x1, y1, x2, y2 = box

            embedding = get_embedding(
                frame,
                x1,
                y1,
                x2,
                y2
            )

            if embedding is None:
                continue

            if track_id in track_to_global:

                global_id = track_to_global[track_id]

                if global_id in known_objects:

                    known_objects[global_id]["last_seen"] = current_time

            else:

                best_match = None

                best_score = -1

                for gid, obj in known_objects.items():

                    if obj["class"] != cls:
                        continue

                    score = cosine_similarity(
                        [embedding],
                        [obj["embedding"]]
                    )[0][0]

                    print(
                        f"Track {track_id} -> "
                        f"Global {gid} : "
                        f"{score:.3f}"
                    )

                    if score > best_score:

                        best_score = score
                        best_match = gid

                    print(
                            f"BEST MATCH = {best_match} "
                            f"SCORE = {best_score:.3f}"
                        )
                if best_score > SIMILARITY_THRESHOLD:

                    global_id = best_match

                    track_to_global[track_id] = global_id

                    known_objects[global_id]["last_seen"] = current_time

                    print(
                        f"REUSED | "
                        f"Track={track_id} "
                        f"Global={global_id} "
                        f"Score={best_score:.3f}"
                    )

                else:

                    global_id = next_global_id

                    next_global_id += 1

                    track_to_global[track_id] = global_id

                    known_objects[global_id] = {
                        "class": cls,
                        "embedding": embedding,
                        "last_seen": current_time
                    }

                    counts[cls] += 1

                    print(
                        f"NEW | "
                        f"Track={track_id} "
                        f"Global={global_id}"
                    )

            known_objects[global_id] = {
                "class": cls,
                "embedding": embedding,
                "last_seen": current_time
            }

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_names[cls]} #{global_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Remotes: {counts[2]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ResNet50 ReID", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()