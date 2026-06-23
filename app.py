#!/usr/bin/env python3
"""
DeepFake MIA - Application Web
Face Swap en temps réel par MIA - La Maison de l'IA

Usage:
    python app.py

Puis ouvrez http://localhost:5000 dans votre navigateur.
"""

import os
import sys
import cv2
import threading
import time
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory, Response

# Configuration
from config import (
    BASE_DIR, STATIC_DIR, TEMPLATES_DIR, FACES_DIR,
    PLAYERS_LEFT, PLAYERS_RIGHT, DEFAULT_OPTIONS, FLASK_CONFIG,
    DET_SIZE, CAMERA_BUFFERSIZE
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Initialisation Flask
# ============================================================

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATES_DIR
)
app.config['SECRET_KEY'] = FLASK_CONFIG['SECRET_KEY']

# ============================================================
# État de l'application
# ============================================================

app_state = {
    "selected_player": None,
    "is_running": False,
    "options": DEFAULT_OPTIONS.copy(),
    "source_face": None,
    "camera": None,
    "camera_lock": threading.Lock()
}

# Mapping des noms d'options frontend (camelCase) -> backend (snake_case)
OPTION_MAP = {
    "mouthMask": "mouth_mask",
    "faceEnhancer": "face_enhancer",
    "showFps": "show_fps",
    "manyFaces": "many_faces",
}

# ============================================================
# Initialisation des modules IA
# ============================================================

def check_gpu_availability():
    """Vérifie la disponibilité du GPU"""
    gpu_info = {
        "cuda_available": False,
        "cudnn_available": False,
        "gpu_name": None,
        "onnx_providers": []
    }
    
    # Vérifier PyTorch CUDA
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if gpu_info["cuda_available"]:
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["cudnn_available"] = torch.backends.cudnn.is_available()
    except Exception:
        pass
    
    # Vérifier ONNX Runtime providers
    try:
        import onnxruntime as ort
        gpu_info["onnx_providers"] = ort.get_available_providers()
    except Exception:
        pass
    
    return gpu_info

def init_ai_modules():
    """Initialise les modules d'IA au démarrage"""
    try:
        import core.globals
        
        # Vérifier le GPU
        gpu_info = check_gpu_availability()
        
        print("\n" + "="*60)
        print("🖥️  CONFIGURATION GPU")
        print("="*60)
        
        if gpu_info["cuda_available"]:
            print(f"✅ CUDA disponible: {gpu_info['gpu_name']}")
            print(f"✅ cuDNN disponible: {gpu_info['cudnn_available']}")
        else:
            print("❌ CUDA non disponible - Mode CPU")
            print("   Pour activer le GPU, installez:")
            print("   - CUDA Toolkit 12.x")
            print("   - cuDNN 9.x (sudo apt install libcudnn9-cuda-12)")
        
        print(f"📦 ONNX Providers: {gpu_info['onnx_providers']}")
        print("="*60 + "\n")
        
        # Configuration des execution providers selon disponibilité
        if 'CUDAExecutionProvider' in gpu_info['onnx_providers']:
            core.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Mode GPU activé (CUDA)")
        else:
            core.globals.execution_providers = ['CPUExecutionProvider']
            logger.info("Mode CPU activé")

        # Taille de détection (levier de performance, voir config.DET_SIZE)
        core.globals.det_size = DET_SIZE
        logger.info(f"Taille de détection des visages: {DET_SIZE}x{DET_SIZE}")
        
        logger.info("Modules IA initialisés avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des modules IA: {e}")
        return False

# ============================================================
# Gestion de la caméra et du face swap
# ============================================================

def get_camera():
    """Obtient ou crée la capture vidéo"""
    with app_state["camera_lock"]:
        if app_state["camera"] is None or not app_state["camera"].isOpened():
            app_state["camera"] = cv2.VideoCapture(0)
            if app_state["camera"].isOpened():
                app_state["camera"].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                app_state["camera"].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                app_state["camera"].set(cv2.CAP_PROP_FPS, 30)
                # Latence minimale : ne pas accumuler de frames en retard
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    app_state["camera"].set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFERSIZE)
                logger.info("Caméra initialisée")
        return app_state["camera"]

def release_camera():
    """Libère la caméra"""
    with app_state["camera_lock"]:
        if app_state["camera"] is not None:
            app_state["camera"].release()
            app_state["camera"] = None
            logger.info("Caméra libérée")

def load_source_face(player_id: str):
    """Charge le visage source depuis l'image du joueur"""
    try:
        from core.face_analyser import extract_face_from_image
        
        face_path = os.path.join(FACES_DIR, f"{player_id}.png")
        if not os.path.exists(face_path):
            logger.error(f"Image non trouvée: {face_path}")
            return None
        
        source_face = extract_face_from_image(face_path)
        if source_face is None:
            logger.error(f"Aucun visage détecté dans: {face_path}")
            return None
        
        logger.info(f"Visage source chargé: {player_id}")
        return source_face
    except Exception as e:
        logger.error(f"Erreur lors du chargement du visage: {e}")
        return None

def process_frame_with_swap(frame, source_face):
    """Applique le face swap sur une frame"""
    if frame is None or source_face is None:
        return frame
    
    try:
        from core.processors.frame.face_swapper import process_frame
        import core.globals
        
        # Appliquer le face swap
        processed = process_frame(source_face, frame)
        
        # Appliquer l'enhancer si activé
        if app_state["options"].get("face_enhancer", False):
            try:
                from core.processors.frame.face_enhancer import process_frame_v2, is_available
                if is_available():
                    processed = process_frame_v2(processed)
            except Exception as e:
                pass  # Face enhancer non disponible, continuer sans
        
        return processed
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        return frame

def generate_frames():
    """Générateur de frames pour le streaming vidéo"""
    camera = get_camera()
    
    if camera is None or not camera.isOpened():
        logger.error("Impossible d'ouvrir la caméra")
        return
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    while app_state["is_running"]:
        success, frame = camera.read()
        if not success:
            logger.warning("Impossible de lire la frame")
            break
        
        # Flip horizontal pour effet miroir
        frame = cv2.flip(frame, 1)
        
        # Appliquer le face swap si un visage source est chargé
        if app_state["source_face"] is not None:
            frame = process_frame_with_swap(frame, app_state["source_face"])
        
        # Calcul et affichage du FPS
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        if app_state["options"].get("show_fps", False):
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    release_camera()

# ============================================================
# Routes principales
# ============================================================

@app.route('/')
def index():
    """Page principale de l'application"""
    return render_template(
        'index.html',
        players_left=PLAYERS_LEFT,
        players_right=PLAYERS_RIGHT
    )

@app.route('/static/faces/<filename>')
def serve_face(filename):
    """Servir les images de visages"""
    return send_from_directory(FACES_DIR, filename)

@app.route('/static/images/<filename>')
def serve_image(filename):
    """Servir les images (logo, fond)"""
    return send_from_directory(os.path.join(STATIC_DIR, 'images'), filename)

@app.route('/static/css/<filename>')
def serve_css(filename):
    """Servir les fichiers CSS"""
    return send_from_directory(os.path.join(STATIC_DIR, 'css'), filename)

@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming vidéo"""
    if not app_state["is_running"]:
        return "Streaming non démarré", 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ============================================================
# API Endpoints
# ============================================================

@app.route('/api/select_face', methods=['POST'])
def api_select_face():
    """Sélectionner un visage pour le deepfake"""
    data = request.get_json()
    player_id = data.get('player')
    
    if not player_id:
        return jsonify({"success": False, "error": "Player ID requis"}), 400
    
    app_state["selected_player"] = player_id
    
    # Charger le visage source
    source_face = load_source_face(player_id)
    if source_face is not None:
        app_state["source_face"] = source_face
        return jsonify({
            "success": True,
            "player": player_id,
            "message": f"Visage {player_id} chargé et prêt"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Impossible de charger le visage {player_id}"
        }), 400

@app.route('/api/start', methods=['POST'])
def api_start():
    """Démarrer le deepfake"""
    if not app_state["selected_player"]:
        return jsonify({"success": False, "error": "Aucun visage sélectionné"}), 400
    
    if app_state["source_face"] is None:
        # Tenter de recharger le visage
        source_face = load_source_face(app_state["selected_player"])
        if source_face is None:
            return jsonify({"success": False, "error": "Impossible de charger le visage source"}), 400
        app_state["source_face"] = source_face
    
    # Mettre à jour les options (traduction camelCase front -> snake_case interne)
    data = request.get_json() or {}
    if data.get('options'):
        for key, value in data['options'].items():
            backend_key = OPTION_MAP.get(key, key)
            if backend_key in app_state["options"]:
                app_state["options"][backend_key] = value

    # Mettre à jour les globals
    import core.globals
    core.globals.many_faces = app_state["options"].get("many_faces", False)
    core.globals.mouth_mask = app_state["options"].get("mouth_mask", False)

    app_state["is_running"] = True
    
    logger.info(f"DeepFake démarré avec visage: {app_state['selected_player']}")
    
    return jsonify({
        "success": True,
        "message": "DeepFake démarré",
        "state": {
            "selected_player": app_state["selected_player"],
            "is_running": app_state["is_running"],
            "options": app_state["options"]
        }
    })

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Arrêter le deepfake"""
    app_state["is_running"] = False
    release_camera()
    
    logger.info("DeepFake arrêté")
    
    return jsonify({
        "success": True,
        "message": "DeepFake arrêté"
    })

@app.route('/api/option', methods=['POST'])
def api_option():
    """Mettre à jour une option"""
    data = request.get_json()
    option = data.get('option')
    value = data.get('value')

    backend_option = OPTION_MAP.get(option, option)
    
    if backend_option not in app_state["options"]:
        return jsonify({"success": False, "error": f"Option inconnue: {option}"}), 400
    
    app_state["options"][backend_option] = value
    
    # Mettre à jour les globals si nécessaire
    import core.globals
    if backend_option == "many_faces":
        core.globals.many_faces = value
    elif backend_option == "mouth_mask":
        core.globals.mouth_mask = value

    logger.info(f"Option mise à jour: {backend_option} = {value}")
    
    return jsonify({
        "success": True,
        "option": backend_option,
        "value": value
    })

@app.route('/api/status')
def api_status():
    """Obtenir le statut actuel"""
    return jsonify({
        "selected_player": app_state["selected_player"],
        "is_running": app_state["is_running"],
        "options": app_state["options"],
        "face_loaded": app_state["source_face"] is not None
    })

@app.route('/api/players')
def api_players():
    """Obtenir la liste des joueurs"""
    return jsonify({
        "left": PLAYERS_LEFT,
        "right": PLAYERS_RIGHT
    })

# ============================================================
# Fonctions utilitaires
# ============================================================

def get_available_cameras():
    """Obtenir la liste des caméras disponibles"""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({"id": str(i), "name": f"Caméra {i + 1}"})
            cap.release()
    return cameras if cameras else [{"id": "0", "name": "Caméra par défaut"}]

# ============================================================
# Point d'entrée
# ============================================================

def main():
    """Point d'entrée principal"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🎭  DeepFake MIA - La Maison de l'IA                     ║
    ║                                                              ║
    ║     Interface Web pour Face Swap en temps réel               ║
    ║                                                              ║
    ║     📍 Ouvrez votre navigateur :                             ║
    ║        http://localhost:5000                                 ║
    ║                                                              ║
    ║     ⏹  Appuyez sur Ctrl+C pour arrêter                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialiser les modules IA
    init_ai_modules()
    
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=False,  # Désactiver debug pour éviter les problèmes avec la caméra
        threaded=True
    )

if __name__ == '__main__':
    main()
