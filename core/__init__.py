"""
DeepFake MIA - Core Module
Logique métier pour le face swap en temps réel
"""

from . import globals
from .face_analyser import get_one_face, get_many_faces
from .video_capture import VideoCapturer

__all__ = [
    'globals',
    'get_one_face',
    'get_many_faces',
    'VideoCapturer'
]
