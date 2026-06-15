"""
DeepFake MIA - Variables globales du pipeline IA

Ces variables sont lues directement par les modules de core/ (face_analyser,
face_swapper). app.py les synchronise depuis app_state lors du démarrage du
swap et des changements d'options.
"""

from typing import List

# Execution providers ONNX Runtime — défini par init_ai_modules() au démarrage
execution_providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Taille de détection des visages — synchronisée depuis config.DET_SIZE au démarrage
det_size = 640

# Options de swap (synchronisées depuis app_state dans app.py)
many_faces = False
mouth_mask = False
preserve_skin_tone = False

# Paramètres du masque de bouche — lus par face_swapper.create_lower_mouth_mask
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1
