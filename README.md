# Application desktop — Prog.Lin

Application **Python / Tkinter** pour la régression linéaire, le clustering, ARIMA, forêts aléatoires, ACP, validation croisée, etc.

**Dépôt :** [github.com/HOUSSAM051/application-desktop-](https://github.com/HOUSSAM051/application-desktop-)

## Prérequis

- Python 3.10+ recommandé
- Windows (Tkinter inclus avec l’installateur officiel Python)

## Installation

```powershell
cd chemin\vers\Prog.Lin
python -m pip install -r requirements.txt
```

## Lancer l’application

```powershell
python main.py
```

## Build exécutable (PyInstaller)

Exemple pour la cible principale :

```powershell
pyinstaller main.spec
```

L’exécutable se trouve dans le dossier `dist\` (après build). Les fichiers `*.spec` référencent `logo.ico` à la racine du projet.
