
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

CONFIG = {
    "films": {
        "mots": ["king", "queen", "love", "war", "family", "murder"],
        "paires": [
            ("king", "queen"),
            ("love", "romance"),
            ("war", "battle"),
            ("murder", "killer"),
            ("family", "mother")
        ]
    },
    "amazon": {
        "mots": ["phone", "battery", "screen", "camera", "charger", "case"],
        "paires": [
            ("phone", "smartphone"),
            ("battery", "charge"),
            ("screen", "display"),
            ("camera", "picture"),
            ("charger", "cable")
        ]
    }
}

similarites = []
voisins = []

for dataset in ["films", "amazon"]:
    for architecture in ["cbow", "skipgram"]:

        nom = f"{dataset}_{architecture}"
        print(f"\nEvaluation de {nom}...", flush=True)

        modele = Word2Vec.load(str(MODELS / f"{nom}.model"))
        wv = modele.wv

        for mot in CONFIG[dataset]["mots"]:
            if mot not in wv:
                print(f"Mot absent : {mot}")
                continue

            proches = wv.most_similar(mot, topn=10)

            print(f"\n{mot} :", proches[:5])

            for rang, (voisin, score) in enumerate(proches, 1):
                voisins.append({
                    "modele": nom,
                    "mot": mot,
                    "rang": rang,
                    "voisin": voisin,
                    "similarite": round(float(score), 4)
                })

        for mot1, mot2 in CONFIG[dataset]["paires"]:
            if mot1 in wv and mot2 in wv:
                score = float(wv.similarity(mot1, mot2))
                similarites.append({
                    "modele": nom,
                    "mot1": mot1,
                    "mot2": mot2,
                    "similarite": round(score, 4)
                })

        # Visualisation PCA des mots et de leurs voisins
        selection = []

        for mot in CONFIG[dataset]["mots"]:
            if mot in wv:
                selection.append(mot)
                selection.extend(
                    m for m, _ in wv.most_similar(mot, topn=3)
                )

        selection = list(dict.fromkeys(selection))

        if len(selection) >= 3:
            vecteurs = np.array([wv[m] for m in selection])
            points = PCA(n_components=2).fit_transform(vecteurs)

            plt.figure(figsize=(12, 8))
            plt.scatter(points[:, 0], points[:, 1])

            for i, mot in enumerate(selection):
                plt.annotate(
                    mot,
                    (points[i, 0], points[i, 1]),
                    fontsize=9
                )

            plt.title(f"Word2Vec - PCA - {nom}")
            plt.xlabel("Composante principale 1")
            plt.ylabel("Composante principale 2")
            plt.tight_layout()
            plt.savefig(RESULTS / f"pca_{nom}.png", dpi=180)
            plt.close()

def enregistrer(nom, lignes):
    if not lignes:
        return
    with open(RESULTS / nom, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=lignes[0].keys())
        writer.writeheader()
        writer.writerows(lignes)

enregistrer("similarites.csv", similarites)
enregistrer("mots_similaires.csv", voisins)

print("\nEVALUATION TERMINEE")
print("Fichiers sauvegardes dans results/")
