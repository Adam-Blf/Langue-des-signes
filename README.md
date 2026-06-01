# Langue des Signes · Reconnaissance Multilingue par Computer Vision

<!-- adam-badges:start -->
[![commits](https://img.shields.io/github/commit-activity/t/Adam-Blf/Langue-des-signes?color=001329&label=commits&style=flat-square)](https://github.com/Adam-Blf/Langue-des-signes/commits) [![visites](https://hits.sh/github.com/Adam-Blf/Langue-des-signes.svg?style=flat-square&label=visites&color=001329)](https://hits.sh/github.com/Adam-Blf/Langue-des-signes/) [![last commit](https://img.shields.io/github/last-commit/Adam-Blf/Langue-des-signes?color=D4A437&style=flat-square&label=dernier%20push)](https://github.com/Adam-Blf/Langue-des-signes/commits) [![top language](https://img.shields.io/github/languages/top/Adam-Blf/Langue-des-signes?style=flat-square)](https://github.com/Adam-Blf/Langue-des-signes) [![license](https://img.shields.io/github/license/Adam-Blf/Langue-des-signes?style=flat-square&color=D4A437)](LICENSE)
<!-- adam-badges:end -->


[![EFREI Paris](https://img.shields.io/badge/EFREI-Paris-005CA9?style=flat-square&labelColor=000000)](https://www.efrei.fr/)

![Status](https://img.shields.io/badge/status-production-brightgreen)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MIT](https://img.shields.io/badge/license-MIT-blue)

Reconnaissance en temps reel des alphabets de langues des signes via webcam. MediaPipe Hands pour l'extraction des 21 landmarks de la main, classificateur ML entraine sur les coordonnees normalisees. Sept langues supportees.

## Architecture

```mermaid
flowchart TB
    A["Webcam<br/>flux video · OpenCV cv2.VideoCapture"]
    B["lsf_model.py<br/>MediaPipe Hands · 21 landmarks 3D"]
    C["Normalisation<br/>bounding box · features geometriques"]
    D["machine_learning/train_model.py<br/>RandomForest scikit-learn"]
    E["model.p<br/>modele serialise pickle"]
    F["detection_pipeline.py<br/>capture · extraction · prediction"]
    G["gui_main.py<br/>interface desktop Tkinter"]
    H["streamlit_app.py<br/>interface web Streamlit"]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    C --> F
    F --> G
    F --> H
```

## Contexte

Projet personnel orientation Computer Vision et accessibilite. Demontre le pipeline complet · capture video temps reel, extraction de features geometriques, classification supervisee, interface Streamlit et desktop Tkinter.

## Langues supportees

Francais (LSF), Anglais (ASL), Espagnol, Italien, Portugais, Russe, Allemand, Turc.

## Methode

- **Capture** · flux webcam via OpenCV (`cv2.VideoCapture`)
- **Extraction** · MediaPipe Hands · 21 landmarks 3D par main detectee, normalisation par bounding box
- **Dataset** · collecte de frames par lettre (script `generate_synthetic_data.py` + collecte manuelle), stockage CSV
- **Modele** · RandomForestClassifier scikit-learn entraine sur les coordonnees normalisees (`train_model.py`)
- **Inference** · `detection_pipeline.py` orchestre capture, extraction, prediction
- **Interfaces** · app desktop Tkinter (`gui_main.py`) et app web Streamlit (`streamlit_app.py`)

## Stack

- **Langage** · Python 3.11
- **Vision** · OpenCV, MediaPipe (solutions.hands)
- **ML** · scikit-learn (RandomForestClassifier), pickle pour la serialisation
- **UI** · Tkinter (desktop), Streamlit (web)
- **Deploiement** · render.yaml configure pour Streamlit Cloud / Render

## Structure

```
Langue-des-signes/
├── lsf_model.py              # Wrapper MediaPipe Hands
├── detection_pipeline.py     # Pipeline capture, extraction, prediction
├── letters_conditions.py     # Regles geometriques par lettre
├── machine_learning/
│   ├── train_model.py        # Entrainement RandomForest
│   ├── generate_synthetic_data.py
│   ├── data.csv              # Dataset features + labels
│   ├── model.p               # Modele serialise
│   └── confusion_matrix.png  # Evaluation
├── app.py                    # Entrypoint principal
├── gui_main.py               # Interface Tkinter desktop
└── streamlit_app.py          # Interface web
```

## Resultats

- Detection robuste en temps reel (>= 20 FPS sur CPU standard)
- Matrice de confusion visualisee dans `machine_learning/confusion_matrix.png`
- Fonctionnement offline apres telechargement du modele MediaPipe

## Reproduction

```bash
git clone https://github.com/Adam-Blf/Langue-des-signes
cd Langue-des-signes
pip install -r requirements.txt

# Desktop (OpenCV window)
python app.py

# Web (Streamlit)
streamlit run streamlit_app.py
```

## Licence

MIT

---

<p align="center">
  <sub>Par <a href="https://adam.beloucif.com">Adam Beloucif</a> · Data Engineer & Fullstack Developer · <a href="https://github.com/Adam-Blf">GitHub</a> · <a href="https://www.linkedin.com/in/adambeloucif/">LinkedIn</a></sub>
</p>