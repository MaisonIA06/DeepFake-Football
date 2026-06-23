import os
from typing import Any
import insightface

import cv2
import numpy as np
import core.globals
from core.typing import Frame

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l', 
            providers=core.globals.execution_providers
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    """Détecte et retourne un seul visage dans la frame"""
    if frame is None:
        return None
    try:
        faces = get_face_analyser().get(frame)
        if faces:
            return min(faces, key=lambda x: x.bbox[0])
    except (ValueError, Exception):
        pass
    return None


def get_many_faces(frame: Frame) -> Any:
    """Détecte et retourne tous les visages dans la frame"""
    if frame is None:
        return None
    try:
        return get_face_analyser().get(frame)
    except (IndexError, Exception):
        return None


def extract_face_from_image(image_path: str) -> Any:
    """Extrait le visage d'une image source"""
    if not os.path.exists(image_path):
        return None
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return get_one_face(frame)
    except Exception:
        return None
