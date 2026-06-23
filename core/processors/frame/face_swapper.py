from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import core.globals
import logging
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            logging.info(f"Chargement du modèle face swapper: {model_path}")
            if not os.path.exists(model_path):
                logging.error(f"❌ Model not found: {model_path}")
                return None
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path, providers=core.globals.execution_providers
                )
                logging.info(f"✅ Face swapper chargé avec succès")
            except Exception as e:
                logging.error(f"❌ Erreur chargement face swapper: {e}")
                return None
    return FACE_SWAPPER


def swap_face(source_face: Any, target_face: Any, temp_frame: np.ndarray) -> np.ndarray:
    """Effectue le swap de visage entre source et target"""
    face_swapper = get_face_swapper()
    if face_swapper is None:
        logging.warning("Face swapper non disponible, retour frame originale")
        return temp_frame

    try:
        logging.debug(f"Swap en cours - Frame shape: {temp_frame.shape}")
        # Apply the face swap
        swapped_frame = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )
        logging.debug(f"Swap effectué - Result shape: {swapped_frame.shape}")

        if core.globals.mouth_mask:
            # Create a mask for the target face
            face_mask = create_face_mask(target_face, temp_frame)

            # Create the mouth mask
            mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
                create_lower_mouth_mask(target_face, temp_frame)
            )

            # Apply the mouth area
            swapped_frame = apply_mouth_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
            )

        return swapped_frame
    except Exception as e:
        logging.error(f"Erreur lors du swap: {e}")
        return temp_frame


def process_frame(source_face: Any, temp_frame: np.ndarray) -> np.ndarray:
    """Traite une frame avec face swap"""
    from core.face_analyser import get_one_face, get_many_faces
    
    if source_face is None:
        return temp_frame

    if core.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if target_face:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    
    return temp_frame


def create_lower_mouth_mask(face: Any, frame: np.ndarray) -> tuple:
    """Crée un masque pour la bouche inférieure"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    min_x, min_y, max_x, max_y = 0, 0, 1, 1
    
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        lower_lip_order = [
            65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        center = np.mean(lower_lip_landmarks, axis=0)
        expansion_factor = 1 + core.globals.mask_down_size
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        toplip_indices = [20, 0, 1, 2, 3, 4, 5]
        toplip_extension = core.globals.mask_size * 0.5
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        chin_indices = [11, 12, 13, 14, 15, 16]
        chin_extension = 2 * 0.2
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        padding = int((max_x - min_x) * 0.1)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)
        mask[min_y:max_y, min_x:max_x] = mask_roi
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    """Applique le masque de bouche"""
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or mouth_polygon is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        feather_amount = min(
            30,
            box_width // core.globals.mask_feather_ratio,
            box_height // core.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        logging.debug(f"Erreur lors de l'application du masque de bouche: {e}")

    return frame


def create_face_mask(face: Any, frame: np.ndarray) -> np.ndarray:
    """Crée un masque pour le visage"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks = landmarks.astype(np.int32)

        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye_brow = landmarks[43:51]
        left_eye_brow = landmarks[97:105]

        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)

        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[::-1],
                [forehead_right],
            ]
        )

        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )

        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)
        cv2.fillConvexPoly(mask, hull_padded, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Applique le transfert de couleur de target vers source"""
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
