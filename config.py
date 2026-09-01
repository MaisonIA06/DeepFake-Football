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
# Thème de l'évènement
# ============================================================
# Le projet est réutilisé d'un évènement à l'autre : seuls le THEME
# ci-dessous et la galerie PLAYERS changent (avec les images dans
# static/faces/). Le reste de l'application est thème-agnostique.

THEME = {
    "event": "Gastronomie",                          # Nom du thème de l'évènement
    "title": "DeepFake",                             # Titre affiché dans le header
    "subtitle": "Prenez le visage d'un grand chef",  # Sous-titre du header
    "panel_title": "Choisissez un chef",             # Titre des panneaux de sélection
    "placeholder_icon": "🍳",                        # Icône du cadre caméra avant démarrage
}

# ============================================================
# Galerie de visages (thème courant : Gastronomie)
# ============================================================
# Convention : chaque entrée a son image static/faces/<id>.png
# (l'id doit correspondre exactement au nom de fichier sans extension).

PLAYERS = [
    {"id": "auguste-escoffier", "name": "Auguste Escoffier", "position": "left"},
    {"id": "joel-robuchon", "name": "Joël Robuchon", "position": "left"},
    {"id": "philippe-etchebest", "name": "Philippe Etchebest", "position": "left"},
    {"id": "thierry-marx", "name": "Thierry Marx", "position": "left"},
    {"id": "alfredo-linguini", "name": "Alfredo Linguini", "position": "left"},
    {"id": "anne-sophie-pic", "name": "Anne-Sophie Pic", "position": "right"},
    {"id": "virginie-basselot", "name": "Virginie Basselot", "position": "right"},
    {"id": "maite", "name": "Maïté", "position": "right"},
    {"id": "louis-camille-maillard", "name": "Louis-Camille Maillard", "position": "right"},
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
