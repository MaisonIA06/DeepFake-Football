# 🎭 DeepFake MIA

**Face Swap en temps réel** - Interface Web moderne développée par **MIA - La Maison de l'IA**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Description

DeepFake MIA est une application web permettant d'effectuer du **face swap en temps réel** via webcam. L'interface moderne et intuitive permet de sélectionner facilement un visage source et d'appliquer le deepfake instantanément.

### ✨ Fonctionnalités

- 🎥 **Face Swap temps réel** via webcam
- 👤 **2 visages pré-configurés** (Kylian Mbappé & Didier Deschamps)
- 🎨 **Interface web moderne** (responsive)
- 🚀 **Support GPU NVIDIA** (CUDA + cuDNN)
- ⚙️ **Options avancées** :
  - Mouth Mask (préserve la bouche originale)
  - Face Enhancer (amélioration qualité GFPGAN)
  - Many Faces (multi-visages)
  - Préserver mon teint (garde votre carnation tout en prenant les traits)
  - Affichage FPS

---

## 🏗️ Structure du Projet

```
DeepFake-MIA/
├── app.py                    # 🚀 Point d'entrée principal (Flask + caméra + streaming)
├── config.py                 # ⚙️ Configuration globale
├── requirements.txt          # 📦 Dépendances Python
├── README.md                 # 📖 Documentation
├── CLAUDE.md                 # 🤖 Guide pour Claude Code
├── LICENSE                   # 📄 Licence
│
├── core/                     # 🧠 Logique métier (pipeline IA)
│   ├── __init__.py
│   ├── globals.py            # Variables globales du pipeline
│   ├── typing.py             # Types partagés (Face, Frame)
│   ├── face_analyser.py      # Détection de visage (InsightFace)
│   └── processors/frame/     # Processeurs de frame
│       ├── face_swapper.py   # Face swap + mouth mask + préservation du teint
│       └── face_enhancer.py  # Amélioration GFPGAN (optionnel)
│
├── models/                   # 🤖 Modèles IA
│   ├── inswapper_128_fp16.onnx
│   ├── GFPGANv1.4.pth
│   └── (autres modèles)
│
├── gfpgan/                   # 🎨 Poids GFPGAN
│   └── weights/
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
│
├── static/                   # 🎨 Assets web
│   ├── css/
│   │   └── style.css
│   ├── images/
│   │   ├── MIA_Assets12.jpg  # Fond terracotta
│   │   └── MIA_Blanc.png     # Logo MIA
│   └── faces/
│       ├── MBAPPE.png        # Visages disponibles (<ID>.png)
│       └── DESCHAMPS.png
│
└── templates/                # 📄 Templates HTML
    └── index.html
```

---

## 🚀 Installation

### Prérequis

- **Python 3.10+**
- **GPU NVIDIA** (recommandé) avec drivers récents
- **Webcam**

### Installation Standard (CPU)

```bash
# 1. Cloner le projet
git clone https://github.com/MaisonIA06/DeepFake.git
cd DeepFake

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Installation GPU NVIDIA (Recommandé)

Pour de meilleures performances avec un GPU NVIDIA :

```bash
# 1. Activer l'environnement virtuel
source venv/bin/activate

# 2. Installer PyTorch avec CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Vérifier l'installation GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Prérequis GPU :**
- CUDA Toolkit 12.x installé
- Drivers NVIDIA à jour (`nvidia-smi` doit fonctionner)

### Télécharger les modèles

Placez les fichiers suivants dans le dossier `models/` :

| Modèle | Téléchargement |
|--------|----------------|
| `inswapper_128_fp16.onnx` | [HuggingFace](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx) |
| `GFPGANv1.4.pth` | [GitHub](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) |

---

## 🎮 Utilisation

### Lancer l'application

```bash
source venv/bin/activate
python app.py
```

Au démarrage, vous verrez le diagnostic GPU :
```
============================================================
🖥️  CONFIGURATION GPU
============================================================
✅ CUDA disponible: NVIDIA GeForce RTX 2080 Ti
✅ cuDNN disponible: True
📦 ONNX Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
============================================================
```

### Accéder à l'interface

Ouvrez votre navigateur :
```
http://localhost:5000
```

### Comment utiliser

1. **Sélectionnez un visage** dans les panels gauche ou droit
2. Attendez le message **"Visage prêt"**
3. **Cliquez sur "Démarrer DeepFake"**
4. **Profitez** du face swap en temps réel !

---

## ⚙️ Options

| Option | Description | Impact Performance |
|--------|-------------|-------------------|
| **Mouth Mask** | Préserve la bouche originale | ✅ Léger |
| **Face Enhancer** | Améliore la qualité (GFPGAN) | ⚠️ Lourd |
| **Many Faces** | Swap tous les visages détectés | ⚠️ Lourd |
| **Préserver mon teint** | Garde votre couleur de peau (traits du joueur, votre teint) | ✅ Léger |
| **Show FPS** | Affiche les images/seconde | ✅ Aucun |

---

## 🎨 Personnalisation

### Ajouter de nouveaux visages

1. Ajoutez l'image dans `static/faces/` au format **`<ID>.png`** : le nom de fichier doit
   correspondre exactement au champ `id` ci-dessous. Un portrait net (le visage occupant
   une bonne partie de l'image) donne les meilleurs résultats ; l'avatar est cadré vers le haut.
2. Modifiez `config.py` pour ajouter la personne :

```python
PLAYERS = [
    # ... entrées existantes ...
    {"id": "NOUVEAU", "name": "Nouvelle Personne", "position": "left"},  # -> static/faces/NOUVEAU.png
]
```

### Modifier le thème

Éditez `static/css/style.css` pour personnaliser :
- Couleurs (variables CSS)
- Polices
- Espacements

---

## 🔧 Configuration

Modifiez `config.py` pour ajuster :

```python
# Serveur Web
FLASK_CONFIG = {
    "HOST": "0.0.0.0",    # Interface réseau
    "PORT": 5000,         # Port
}

# Performance
DET_SIZE = 320          # Taille de détection des visages : principal levier de FPS
                        #   320 = rapide / 480 = équilibré / 640 = précis (à distance)
CAMERA_BUFFERSIZE = 1   # Tampon caméra : 1 = latence minimale
```

> ℹ️ `DET_SIZE` est lu une fois au démarrage : modifiez-le puis **redémarrez** `python app.py`
> pour qu'il soit pris en compte. Si des visages proches/éloignés ne sont plus détectés,
> remontez à `480` ou `640`.

---

## 🐛 Dépannage

### NumPy 2.x incompatible

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution :**
```bash
pip install "numpy<2.0.0"
```

### CUDA / cuDNN non trouvé

```
libcudnn.so.8: cannot open shared object file
```

**Solutions :**
```bash
# Option 1: Installer cuDNN
sudo apt install libcudnn8  # Ubuntu

# Option 2: Utiliser CPU uniquement
pip uninstall onnxruntime-gpu
pip install onnxruntime
```

### Webcam non détectée

- Vérifiez que la webcam fonctionne : `ls /dev/video*`
- L'application utilise la caméra côté serveur (pas le navigateur)
- Fermez les autres applications utilisant la webcam

### Performance faible

- ✅ Utilisez un **GPU NVIDIA** avec CUDA
- ⚙️ Réduisez **`DET_SIZE`** dans `config.py` (320 = rapide ; principal levier de FPS)
- ⚠️ Désactivez **"Face Enhancer"** (très gourmand)
- 📉 Réduisez la résolution caméra dans `app.py` (640x480 par défaut)

### Segmentation fault

Souvent causé par NumPy 2.x :
```bash
pip install "numpy<2.0.0"
```

---

## 📊 Performance

| Configuration | FPS Attendus |
|---------------|--------------|
| CPU seul | 2-5 FPS |
| GPU GTX 1060 | 10-15 FPS |
| GPU RTX 2080 Ti | 25-35 FPS |
| GPU RTX 3090 | 35-50 FPS |

*Avec Face Enhancer désactivé*

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🏢 Crédits

Développé par **MIA - La Maison de l'IA**

Technologies utilisées :
- [InsightFace](https://github.com/deepinsight/insightface) - Détection et analyse faciale
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Amélioration de visage
- [ONNX Runtime](https://onnxruntime.ai/) - Inférence optimisée

---

## ⚠️ Avertissement

Cette application est destinée à un usage **éducatif et ludique uniquement**. 

L'utilisation de deepfakes doit se faire :
- Avec le **consentement** des personnes concernées
- Dans le **respect de la vie privée**
- Sans intention de **nuire ou tromper**

Les créateurs de cette application déclinent toute responsabilité en cas d'utilisation abusive.
