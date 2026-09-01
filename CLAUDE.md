# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vue d'ensemble

**DeepFake MIA** est une application web Flask de face swap en temps réel via webcam. Le serveur capture la webcam côté serveur (et non navigateur), applique un face swap image par image avec le modèle InsightFace `inswapper_128`, puis diffuse le flux au navigateur via MJPEG (`multipart/x-mixed-replace`). Le visage source est choisi dans une galerie prédéfinie.

**Démonstrateur événementiel re-thématisable** : le projet est réutilisé d'un évènement à l'autre en ne changeant que `THEME` (textes de l'UI : titre, sous-titre, titre des panneaux, icône) et `PLAYERS` (galerie) dans `config.py`, plus les images `static/faces/`. Thème courant : **Gastronomie** (grands chefs de cuisine). Le reste du code est thème-agnostique — ne pas coder en dur de texte lié au thème ailleurs que dans `config.py`.

Le coeur du pipeline (`core/`) est dérivé de Deep-Live-Cam ; `app.py` et `config.py` sont la couche web/UI propre à MIA.

## Commandes

```bash
# Environnement
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# GPU NVIDIA (recommandé) — remplace le torch CPU par la build CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Lancer (dev) — ouvre http://localhost:5000
python app.py

# Lancer (prod Ubuntu) — démarre Flask + navigateur en mode kiosk plein écran
./launch.sh

# Diagnostic GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import onnxruntime as ort; print(ort.get_available_providers())"

# Installer le raccourci bureau (.desktop)
./install-desktop.sh
```

Il n'y a **pas de suite de tests, ni de linter configuré** dans ce dépôt.

## Modèles requis (non versionnés, dans `models/`)

L'app démarre sans eux mais le swap/enhancer échouera silencieusement (frame originale retournée).

| Fichier | Usage | Source |
|---|---|---|
| `models/inswapper_128_fp16.onnx` | Face swap (obligatoire) | HuggingFace `hacksider/deep-live-cam` |
| `models/GFPGANv1.4.pth` | Face enhancer (optionnel) | GitHub TencentARC/GFPGAN |

`buffalo_l` (détection/analyse InsightFace) est téléchargé automatiquement au premier lancement.

## Architecture

### Flux d'une requête de face swap

1. **Sélection** : `POST /api/select_face` → `load_source_face()` → `extract_face_from_image()` extrait l'embedding du visage source depuis `static/faces/<ID>.png` et le stocke dans `app_state["source_face"]`.
2. **Démarrage** : `POST /api/start` met `app_state["is_running"] = True` et synchronise les options vers `core.globals`.
3. **Streaming** : `GET /video_feed` → `generate_frames()` boucle tant que `is_running` : lit la webcam, flip miroir, `process_frame_with_swap()`, encode JPEG, yield MJPEG.
4. **Arrêt** : `POST /api/stop` met `is_running = False` et libère la caméra.

### État global — deux sources de vérité à garder synchronisées

- **`app_state`** (dans `app.py`) : état de la session web (joueur sélectionné, options UI, `source_face`, handle caméra protégé par `camera_lock`). C'est la couche HTTP.
- **`core.globals`** : variables lues directement par le pipeline (`many_faces`, `mouth_mask`, `show_fps`, `execution_providers`, paramètres de masque...). Le pipeline `core/` ne connaît PAS `app_state`.

⚠️ Toute option qui doit affecter le rendu doit être propagée de `app_state["options"]` vers `core.globals` (voir `api_start` et `api_option`). Modifier seulement `app_state` est sans effet sur le pipeline. Le frontend envoie des noms camelCase (`mouthMask`, `faceEnhancer`...) mappés vers snake_case via `option_map` dans `api_option`.

### Pipeline `core/`

- `core/face_analyser.py` : singleton `FACE_ANALYSER` (InsightFace `buffalo_l`). `get_one_face` retourne le visage le plus à gauche ; `get_many_faces` retourne tous les visages.
- `core/processors/frame/face_swapper.py` : singleton `FACE_SWAPPER` thread-safe (`inswapper_128`). `process_frame()` est le point d'entrée du swap. Contient aussi la logique **mouth mask** (préservation de la bouche d'origine via `landmark_2d_106` et transfert de couleur LAB).
- `core/processors/frame/face_enhancer.py` : GFPGAN, **chargé paresseusement et tolérant aux pannes** — `GFPGAN_AVAILABLE` / `FACE_ENHANCER_FAILED` font qu'un échec d'import ou de modèle désactive l'enhancer sans crasher l'app. Toujours vérifier `is_available()` avant usage.
- Les modèles sont des **singletons globaux** instanciés au premier appel et réutilisés (coûteux à charger). `execution_providers` doit être défini dans `core.globals` AVANT le premier appel (fait par `init_ai_modules()` au démarrage).

### Détection GPU/CPU

`init_ai_modules()` (appelé au démarrage de `main()`) sonde PyTorch CUDA + ONNX Runtime providers, affiche un diagnostic, puis fixe `core.globals.execution_providers` à `['CUDAExecutionProvider', 'CPUExecutionProvider']` ou `['CPUExecutionProvider']`. Ce choix conditionne tous les modèles chargés ensuite.

## Conventions importantes

- **Ajouter un visage** : déposer `static/faces/<id>.png` (PNG obligatoire — le chargement construit `<id>.png` en dur et OpenCV ne lit pas l'AVIF ; convertir avant) ET ajouter une entrée dans `PLAYERS` (`config.py`) avec `position: "left"` ou `"right"`. L'`id` doit correspondre exactement au nom de fichier sans extension. ⚠️ Vérifier la détection après ajout : les visages non humains (mascottes, animaux type Rémy de Ratatouille) ne sont pas détectés par InsightFace et sont inutilisables ; les personnages animés humains (ex. Alfredo Linguini) passent.
- **Caméra** : ouverte côté serveur en 640×480@30 sur l'index `0`. Une seule session caméra à la fois (verrou `camera_lock`).
- **NumPy** : épinglé `<2.0.0` — NumPy 2.x casse les binaires compilés (insightface/opencv) et provoque des segfaults. Ne pas relâcher cette contrainte.
- **Debug Flask** : `app.run(debug=False)` est forcé dans `main()` même si `config.FLASK_CONFIG["DEBUG"]` vaut `True` — le reloader du mode debug entre en conflit avec la caméra.
- **Erreurs du pipeline** : les fonctions de swap/enhancement avalent les exceptions et retournent la frame d'origine plutôt que de crasher le flux vidéo. Un swap qui ne s'applique pas est souvent un modèle manquant ou un visage non détecté, à chercher dans les logs `logging`, pas dans une stack trace.
