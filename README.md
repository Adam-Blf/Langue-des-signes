# Hand Tracking - Langue des Signes

Application temps réel de détection de lettres en langue des signes française. Le projet combine des règles heuristiques, un modèle RandomForest entraîné sur des points de main Mediapipe et une interface Tkinter légère. Chaque module est commenté pour que n’importe qui puisse reprendre le développement.

## Fonctionnalités
- Aperçu caméra en direct avec effet miroir et panneau latéral détaillant la lettre détectée, sa source (règles ou ML) et la confiance.
- Pipeline hybride ajustable : seuil de confiance ML, nombre de votes nécessaires, activation/désactivation indépendante des règles et du modèle.
- Lissage temporel afin de limiter le scintillement et conserver la dernière lettre stable en cas de perte de trames.
- Tampon de transcription avec boutons d’édition (espace, suppression, effacement) et historique défilant des lettres stabilisées.
- Outils CLI modernisés pour collecter des données, réentraîner le modèle, produire un rapport JSON et lancer une validation croisée.
- Fichier `lsf_detector.log` généré automatiquement pour simplifier le diagnostic.
- Spécification PyInstaller prête à l’emploi et script PowerShell pour créer un exécutable Windows autonome.

## Prise en main rapide

```bash
git clone https://github.com/Razane1414/Hand-Tracking---Langue-des-signes.git
cd Hand-Tracking---Langue-des-signes
python -m venv .venv
.venv\Scripts\activate  # PowerShell sur Windows
pip install -r requirements.txt
python gui_main.py
```

Au lancement, la détection démarre automatiquement. La barre latérale indique la lettre stabilisée, la méthode utilisée, la confiance et l’état du pipeline. Ajustez le seuil ML ou désactivez un détecteur pour visualiser immédiatement l’impact.

## Collecte de données et entraînement

```bash
# Collecte avec miroir vidéo, liaisons clavier personnalisées et réinitialisation du CSV
python machine_learning/collect_data.py --letters a b c d e f --overwrite

# Entraînement avec rapport JSON et validation croisée en 5 plis
python machine_learning/train_model.py --cv-folds 5 --report-path machine_learning/model_report.json
```

- Le collecteur affiche les touches disponibles (`A -> lettre`) et n’enregistre que lorsqu’une main est détectée.
- L’entraîneur calcule accuracy, matrice de confusion, rapport de classification, scores de cross-validation et met à jour `machine_learning/model.pkl`.
- Ajoutez `--help` pour découvrir les paramètres (index caméra, seuils Mediapipe, chemins personnalisés, jobs parallèles, etc.).
- L’interface reste fonctionnelle sans modèle : elle se replie sur les règles heuristiques jusqu’à ce qu’un modèle soit entraîné.

## Générer un exécutable (Windows)

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pwsh packaging/build_exe.ps1
```

Le script installe PyInstaller si besoin, reconstruit les dépendances propres (`--clean`) et utilise `packaging/gui_main.spec`. L’exécutable `dist\lsf-detector.exe` est produit ; diffusez tout le dossier `dist\lsf-detector\` pour conserver les ressources (poids Mediapipe, modèle ML, etc.). Personnalisez la spec pour ajouter d’autres fichiers ou changer l’icône.

## Organisation du projet

```
gui_main.py                # Interface Tkinter et réglages du pipeline
detection_pipeline.py      # Fusion règles + ML avec lissage temporel
letters_conditions.py      # Heuristiques de détection classiques
predict_ml.py              # Chargement paresseux du modèle scikit-learn
machine_learning/
    collect_data.py        # Outil de capture interactivé (CLI + logging)
    train_model.py         # Entraînement avec rapports et options avancées
tests/                     # Tests unitaires sur les règles et le pipeline
packaging/                 # Script de build PyInstaller et fichier .spec
```

## Tests

```
pytest
```

Les tests utilisent des landmarks synthétiques : ils s’exécutent très vite et ne nécessitent pas de webcam.

## Dépannage
- **Aucune image caméra** : vérifier qu’aucun autre logiciel n’utilise la webcam. Le programme affichera un message d’erreur clair si l’ouverture échoue.
- **Initialisation Mediapipe échouée** : lancer `dist\lsf-detector.exe` depuis son dossier ou vérifier `lsf_detector.log`. Les modèles Mediapipe doivent rester à côté de l’exécutable.
- **Import manquant** : recréer l’environnement virtuel et réinstaller les dépendances (`pip install -r requirements.txt`).
- **Modèle absent** : le pipeline fonctionne uniquement avec les règles heuristiques. Relancer l’entraînement pour générer `model.pkl`.

## Licence

Le projet conserve la licence du dépôt d’origine. Consultez le dépôt initial pour les termes complets.
