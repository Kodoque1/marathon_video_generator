
# Marathon Reboot Homage Video Generator

Générateur Python de trailer hommage à l'interface sci-fi de Marathon, basé sur :
- grille modulaire stricte
- composition asymétrique et espaces négatifs
- typographie technique
- keylines, brackets, trims, blocs de données
- pipeline d'assemblage ffmpeg
- téléchargement en ligne (vecteurs, polices, musiques)

## Prérequis

- Linux
- Python 3.10+
- `ffmpeg` dans le PATH
- Blender 4.x (pour la couche 3D)

## Installation

```bash
cd /home/kodoque/Documents/marathon_reboot_video_gen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

1. **Blender 3D** :
	 - Génère les frames PNG avec fond transparent.
	 - Commande :
		 ```bash
		 blender --background --python blender_scene.py -- \
				 --output tmp/blender_frames/ --frames 90 --width 1920 --height 1080 --fps 30
		 ```
2. **HUD pycairo** :
	 - Génère les overlays HUD en PNG.
	 - Commande :
		 ```bash
		 python generate_video.py --config config.yaml --hud-layer --hud-output tmp/hud_frames/
		 ```
3. **Composite final** :
	 - Assemble les couches avec ffmpeg.
	 - Commande :
		 ```bash
		 python composite.py --config config.yaml
		 ```

## Utilisation rapide

Pour générer la vidéo complète :
```bash
python generate_video.py --config config.yaml
```
Le résultat sera dans `output/marathon_homage.mp4`.

Pour un composite hybride (Blender + HUD) :
```bash
python composite.py --config config.yaml
```
Le résultat sera dans `output/marathon_hybrid.mp4`.

Options avancées :
- `--preview` : rendu rapide 3s en 854x480
- `--skip-blender` : réutilise les frames Blender existantes
- `--skip-hud` : réutilise les frames HUD existantes

## Configuration

Le fichier `config.yaml` permet d'ajuster :
- résolution, durée, codec vidéo
- palette de couleurs, typographie, grille
- chemins des assets et sorties


## Version v2 : Pipeline Blender-only

La version v2 s'appuie uniquement sur Blender (pas de HUD pycairo, pas de matplotlib). Tous les éléments graphiques (grille, HUD, brackets, Line Art, texte, etc.) sont générés nativement dans Blender via Grease Pencil 3, Curve, TextCurve, et GP3 drawing API.

### Prérequis spécifiques
- Blender 5.x recommandé (GP3 drawing API)
- ffmpeg

### Étapes

1. **Rendu Blender**
	 - Génère les frames PNG avec fond transparent.
	 - Commande :
		 ```bash
		 blender --background --python blender_scene_v2.py -- \
				 --output tmp/blender_v2_frames/ --frames 90 --width 1920 --height 1080 --fps 30
		 ```

2. **Encodage vidéo**
	 - Assemble la séquence PNG en MP4 (H.264, yuv420p) avec ffmpeg.
	 - Commande :
		 ```bash
		 python composite_v2.py --frames-dir tmp/blender_v2_frames/ --output output/marathon_v2.mp4 --fps 30 --crf 18 --preset slow
		 ```
	 - Le résultat sera dans `output/marathon_v2.mp4`.

### Options avancées
- Vous pouvez ajuster le nombre de frames, la résolution, le CRF (qualité), le preset ffmpeg.
- Le script composite_v2.py propose aussi une fonction `render_blender()` pour automatiser le rendu Blender.

### Résumé
- Aucun overlay HUD externe : tout est généré dans Blender.
- Pipeline plus simple, idéale pour tests ou pour une esthétique 100% Blender.

---

Voir `requirements.txt` pour les dépendances Python (utiles pour la version hybride).

## Notes

- Le générateur applique strictement les contraintes de style.
- Les téléchargements en ligne sont best-effort (fallback procédural si échec).
- Vérifiez les licences des ressources téléchargées avant usage commercial.
