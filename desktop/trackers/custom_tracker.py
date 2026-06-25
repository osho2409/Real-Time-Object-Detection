from ultralytics import YOLO
import cv2
import numpy as np
import time
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
# ==========================
# CONFIG
# ==========================

SIMILARITY_THRESHOLD = 0.85  # 0.0 to 1.0
MEMORY_TIMEOUT = 30  # seconds

# ==========================
# MODEL
# ==========================

model = YOLO("best.pt")

# ==========================
# GLOBAL MEMORY
# ==========================

known_objects = {}
track_to_global = {}

next_global_id = 1

counts = {
    0: 0,  # bottle
    1: 0,  # cup
    2: 0   # remote
}

class_names = {
    0: "Bottle",
    1: "Mug",
    2: "Remote"
}

# ==========================
# HELPERS
# ==========================

def get_signature(frame, x1, y1, x2, y2):

    roi = frame[
        max(0, y1):max(y1 + 1, y2),
        max(0, x1):max(x1 + 1, x2)
    ]

    if roi.size == 0:
        return None

    h, w = roi.shape[:2]

    # Center 50% only
    roi = roi[
        h // 4 : 3 * h // 4,
        w // 4 : 3 * w // 4
    ]

    if roi.size == 0:
        return None

    avg_color = np.mean(
        roi.reshape(-1, 3),
        axis=0
    )

    w_box = x2 - x1
    h_box = y2 - y1

    area = w_box * h_box
    aspect_ratio = w_box / max(h_box, 1)

    return {
        "avg_color": avg_color,
        "area": area,
        "aspect_ratio": aspect_ratio
    }


def compute_similarity(sig1, sig2):

    color_dist = np.linalg.norm(
        sig1["avg_color"] -
        sig2["avg_color"]
    )

    color_score = max(
        0,
        1 - (color_dist / 441.67)
    )

    area_score = min(
        sig1["area"],
        sig2["area"]
    ) / max(
        sig1["area"],
        sig2["area"]
    )

    ratio_diff = abs(
        sig1["aspect_ratio"] -
        sig2["aspect_ratio"]
    )

    ratio_score = max(
        0,
        1 - ratio_diff
    )

    score = (
        0.70 * color_score +
        0.20 * area_score +
        0.10 * ratio_score
    )

    return score


# ==========================
# TRACKING
# ==========================

results = model.track(
    source=0,
    tracker="botsort.yaml",
    conf=0.60,
    persist=True,
    stream=True,
    verbose=False
)

prev_time = time.time()

fps_counter = 0
fps_display = 0
fps_update_time = time.time()

for r in results:
    fps_counter += 1

    current_time = time.time()

    if current_time - fps_update_time >= 1:

        fps_display = fps_counter

        fps_counter = 0

        fps_update_time = current_time

    frame = r.orig_img.copy()

    current_time = time.time()

    # --------------------------
    # Remove old memory
    # --------------------------

    expired = []

    for gid, obj in known_objects.items():

        if current_time - obj["last_seen"] > MEMORY_TIMEOUT:
            expired.append(gid)

    for gid in expired:
        del known_objects[gid]

    # --------------------------
    # Process detections
    # --------------------------

    if r.boxes.id is not None:

        ids = r.boxes.id.cpu().numpy().astype(int)
        classes = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

        for track_id, cls, box in zip(ids, classes, boxes):

            x1, y1, x2, y2 = box

            signature = get_signature(
                frame,
                x1, y1, x2, y2
            )
            
            

            if signature is None:
                continue

            # --------------------------------
            # Existing Track
            # --------------------------------

            if track_id in track_to_global:

                global_id = track_to_global[track_id]

                if global_id in known_objects:
                    known_objects[global_id]["last_seen"] = current_time

            else:

                # ----------------------------
                # ReID Match Search
                # ----------------------------

                best_match = None
                best_score = -1

                for gid, obj in known_objects.items():

                    if obj["class"] != cls:
                        continue

                    score = compute_similarity(
                        signature,
                        obj["signature"]
                    )

                    if score > best_score:
                        best_score = score
                        best_match = gid
                        print(f"Track {track_id} -> Global {gid} : {score:.3f}")

                # ----------------------------
                # Reuse Existing Identity
                # ----------------------------

                if best_score > SIMILARITY_THRESHOLD:

                    print(
                        f"REUSED | Track={track_id} "
                        f"Global={best_match} "
                        f"Score={best_score:.3f}"
                    )

                    global_id = best_match

                    track_to_global[track_id] = global_id

                    known_objects[global_id]["last_seen"] = current_time

                # ----------------------------
                # Create New Identity
                # ----------------------------

                else:
                    print(f"NEW | Track={track_id} "f"BestScore={best_score:.3f}")
                    global_id = next_global_id
                    next_global_id += 1

                    track_to_global[track_id] = global_id

                    known_objects[global_id] = {
                        "class": cls,
                        "signature": signature,
                        "last_seen": current_time
                    }

                    counts[cls] += 1

            # Update Signature

           # Update Signature

            if global_id in known_objects:

                old_sig = known_objects[global_id]["signature"]

                #signature["avg_color"] = (
                 #   0.8 * old_sig["avg_color"] +
                  #  0.2 * signature["avg_color"]
                #)

                signature["area"] = (
                    0.8 * old_sig["area"] +
                    0.2 * signature["area"]
                )

                signature["aspect_ratio"] = (
                    0.8 * old_sig["aspect_ratio"] +
                    0.2 * signature["aspect_ratio"]
                )

            known_objects[global_id] = {
                "class": cls,
                "signature": signature,
                "last_seen": current_time
            }

            # Draw

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_names[cls]}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    # --------------------------
    # Counts
    # --------------------------

    frame_width = frame.shape[1]

    x_pos = frame_width - 220

    cv2.putText(
    frame,
    f"FPS: {fps_display}",    
    (x_pos, 160),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
    )

    cv2.putText(
        frame,
        f"Bottles: {counts[0]}",
        (x_pos, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Mugs: {counts[1]}",
        (x_pos, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Remotes: {counts[2]}",
        (x_pos, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    cv2.imshow("Tracking + ReID", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()