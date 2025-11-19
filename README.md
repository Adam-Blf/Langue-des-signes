# Détection Langue des Signes / Sign Language Detection

[🇫🇷 Version Française](#version-française) | [🇬🇧 English Version](#english-version)

---

## <a name="version-française"></a>🇫🇷 Version Française

Application temps réel de détection de lettres en **langue des signes française (LSF)**. Le projet combine des règles heuristiques, un modèle RandomForest entraîné sur des points de main MediaPipe et une interface Tkinter légère. Chaque module est commenté pour faciliter la reprise du développement.

### ✨ Fonctionnalités

- 📹 **Aperçu Caméra Temps Réel** : effet miroir et panneau latéral détaillant la lettre détectée
- 🤖 **Pipeline Hybride** : fusion règles heuristiques + modèle ML avec seuil de confiance ajustable
- 🎯 **Lissage Temporel** : limite le scintillement et conserve la dernière lettre stable
- ✏️ **Tampon de Transcription** : édition (espace, suppression, effacement) et historique
- 🔧 **Outils CLI Modernisés** : collecte données, entraînement, validation croisée, rapports JSON
- 📝 **Logs Automatiques** : fichier `lsf_detector.log` pour diagnostic
- 📦 **Exécutable Windows** : spec PyInstaller et script PowerShell prêts

### 🛠️ Stack Technologique

| Composant | Technologie | Objectif |
|-----------|-------------|----------|
| **GUI Framework** | Tkinter | Interface utilisateur native Python |
| **Détection Main** | MediaPipe Hands | Extraction landmarks main 21 points |
| **ML Model** | scikit-learn RandomForest | Classification lettres LSF |
| **Vision Ordinateur** | OpenCV | Capture webcam et traitement image |
| **Packaging** | PyInstaller | Exécutable Windows autonome |
| **Langage** | Python 3.9+ | Logique applicative |

### 📁 Structure du Projet

```
Langue-des-signes/
├── gui_main.py                # Interface Tkinter et réglages pipeline
├── detection_pipeline.py      # Fusion règles + ML avec lissage temporel
├── letters_conditions.py      # Heuristiques de détection classiques
├── predict_ml.py              # Chargement paresseux modèle scikit-learn
├── requirements.txt           # Dépendances Python
├── machine_learning/
│   ├── collect_data.py        # Outil de capture interactif
│   ├── train_model.py         # Entraînement avec rapports
│   └── data.csv               # Dataset collecté
├── packaging/
│   ├── build_exe.ps1          # Script construction PyInstaller
│   └── gui_main.spec          # Configuration PyInstaller
├── tests/
│   ├── test_detection_pipeline.py
│   └── test_letters_conditions.py
└── README.md
```

### 🚀 Démarrage Rapide

```bash
git clone https://github.com/Razane1414/Hand-Tracking---Langue-des-signes.git
cd Hand-Tracking---Langue-des-signes
python -m venv .venv
.venv\Scripts\activate  # PowerShell Windows
pip install -r requirements.txt
python gui_main.py
```

Au lancement, la détection démarre automatiquement. La barre latérale indique la lettre stabilisée, la méthode utilisée, la confiance et l'état du pipeline.

### 🎯 Collecte de Données et Entraînement

```bash
# Collecte avec miroir vidéo et liaisons clavier
python machine_learning/collect_data.py --letters a b c d e f --overwrite

# Entraînement avec validation croisée 5-fold
python machine_learning/train_model.py --cv-folds 5 --report-path machine_learning/model_report.json
```

- Le collecteur affiche les touches disponibles et n'enregistre que si une main est détectée
- L'entraîneur calcule accuracy, matrice confusion, scores cross-validation
- L'interface fonctionne sans modèle en se repliant sur les règles

### 📦 Générer Exécutable (Windows)

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pwsh packaging/build_exe.ps1
```

L'exécutable `dist\lsf-detector.exe` est produit. Diffusez tout le dossier `dist\lsf-detector\` pour conserver les ressources.

### 🧪 Tests

```bash
pytest
```

Les tests utilisent des landmarks synthétiques et ne nécessitent pas de webcam.

### 🗺️ Feuille de Route

- [ ] Support alphabet complet LSF (26 lettres)
- [ ] Détection mots et phrases
- [ ] Multilangue (ASL, LSQ, etc.)
- [ ] Feedback vocal synthétisé
- [ ] Mode d'apprentissage interactif
- [ ] Support GPU pour inférence
- [ ] Application mobile (iOS/Android)

---

## <a name="english-version"></a>🇬🇧 English Version

Real-time **French Sign Language (LSF)** letter detection application. The project combines heuristic rules, a RandomForest model trained on MediaPipe hand landmarks, and a lightweight Tkinter interface. Each module is commented for easy development continuation.

### ✨ Features

- 📹 **Real-Time Camera Preview**: mirror effect and side panel showing detected letter
- 🤖 **Hybrid Pipeline**: fusion of heuristic rules + ML model with adjustable confidence threshold
- 🎯 **Temporal Smoothing**: reduces flickering and keeps last stable letter
- ✏️ **Transcription Buffer**: editing (space, delete, clear) and scrolling history
- 🔧 **Modern CLI Tools**: data collection, training, cross-validation, JSON reports
- 📝 **Automatic Logs**: `lsf_detector.log` file for diagnostics
- 📦 **Windows Executable**: PyInstaller spec and PowerShell script ready

### 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **GUI Framework** | Tkinter | Native Python user interface |
| **Hand Detection** | MediaPipe Hands | Extract 21-point hand landmarks |
| **ML Model** | scikit-learn RandomForest | LSF letter classification |
| **Computer Vision** | OpenCV | Webcam capture and image processing |
| **Packaging** | PyInstaller | Standalone Windows executable |
| **Language** | Python 3.9+ | Core application logic |

### 📁 Project Structure

```
Langue-des-signes/
├── gui_main.py                # Tkinter interface and pipeline settings
├── detection_pipeline.py      # Rules + ML fusion with temporal smoothing
├── letters_conditions.py      # Classic detection heuristics
├── predict_ml.py              # Lazy loading scikit-learn model
├── requirements.txt           # Python dependencies
├── machine_learning/
│   ├── collect_data.py        # Interactive capture tool
│   ├── train_model.py         # Training with reports
│   └── data.csv               # Collected dataset
├── packaging/
│   ├── build_exe.ps1          # PyInstaller build script
│   └── gui_main.spec          # PyInstaller configuration
├── tests/
│   ├── test_detection_pipeline.py
│   └── test_letters_conditions.py
└── README.md
```

### 🚀 Quick Start

```bash
git clone https://github.com/Razane1414/Hand-Tracking---Langue-des-signes.git
cd Hand-Tracking---Langue-des-signes
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
python gui_main.py
```

At launch, detection starts automatically. The sidebar shows stabilized letter, method used, confidence, and pipeline state.

### 🎯 Data Collection and Training

```bash
# Collection with video mirror and keyboard bindings
python machine_learning/collect_data.py --letters a b c d e f --overwrite

# Training with 5-fold cross-validation
python machine_learning/train_model.py --cv-folds 5 --report-path machine_learning/model_report.json
```

- Collector displays available keys and only records when hand detected
- Trainer calculates accuracy, confusion matrix, cross-validation scores
- Interface works without model by falling back to rules

### 📦 Build Executable (Windows)

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pwsh packaging/build_exe.ps1
```

The executable `dist\lsf-detector.exe` is produced. Distribute the entire `dist\lsf-detector\` folder to preserve resources.

### 🧪 Tests

```bash
pytest
```

Tests use synthetic landmarks and don't require webcam.

### 🗺️ Roadmap

- [ ] Full LSF alphabet support (26 letters)
- [ ] Word and phrase detection
- [ ] Multilingual (ASL, LSQ, etc.)
- [ ] Synthesized voice feedback
- [ ] Interactive learning mode
- [ ] GPU support for inference
- [ ] Mobile app (iOS/Android)

### 📄 License

This project is open source. See LICENSE file for details.

---

**Author**: Razane & Adam Beloucif  
**Repository**: [github.com/Razane1414/Hand-Tracking---Langue-des-signes](https://github.com/Razane1414/Hand-Tracking---Langue-des-signes)

For bug reports or feature requests, open an issue on GitHub.
