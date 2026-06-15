"""
DeepFake MIA - Configuration
Paramètres globaux de l'application
"""

import os

# ============================================================
# Chemins
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
FACES_DIR = os.path.join(STATIC_DIR, 'faces')

# Modèles IA
INSWAPPER_MODEL = os.path.join(MODELS_DIR, 'inswapper_128_fp16.onnx')
GFPGAN_MODEL = os.path.join(MODELS_DIR, 'GFPGANv1.4.pth')
GFPGAN_WEIGHTS_DIR = os.path.join(BASE_DIR, 'gfpgan', 'weights')

# ============================================================
# Joueurs / Visages disponibles
# ============================================================

PLAYERS = [
    {"id": "MBAPPE", "name": "Kylian Mbappé", "position": "left"},
    {"id": "DESCHAMPS", "name": "Didier Deschamps", "position": "right"},
]

PLAYERS_LEFT = [p for p in PLAYERS if p["position"] == "left"]
PLAYERS_RIGHT = [p for p in PLAYERS if p["position"] == "right"]

# ============================================================
# Options par défaut
# ============================================================

DEFAULT_OPTIONS = {
    "mouth_mask": False,
    "face_enhancer": False,
    "show_fps": False,
    "many_faces": False,
    "preserve_skin_tone": False,
}

# ============================================================
# Serveur Web
# ============================================================

FLASK_CONFIG = {
    "SECRET_KEY": "deepfake-mia-2025",
    "DEBUG": True,
    "HOST": "0.0.0.0",
    "PORT": 5000,
}

# ============================================================
# Exécution
# ============================================================

EXECUTION_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
EXECUTION_THREADS = 8
MAX_MEMORY = 8  # GB

# ============================================================
# Performance
# ============================================================

# Taille de l'image de détection des visages (carré, en pixels).
# C'est le principal levier de FPS : plus petit = plus rapide, mais détecte
# moins bien les visages éloignés/petits.
#   320 = rapide  (recommandé webcam, sujet proche de la caméra)
#   480 = équilibré
#   640 = précis  (détection à distance) mais plus lent
DET_SIZE = 320

# Tampon de la caméra. 1 = latence minimale : on traite toujours la frame la
# plus récente au lieu de vider une file d'images en retard.
CAMERA_BUFFERSIZE = 1
