
import json
import csv
import time
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
from gensim.models import Word2Vec

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

TESTS = [
    ("vector_size", 50),
    ("vector_size", 100),
    ("vector_size", 200),
    ("window", 2),
    ("window", 5),
    ("window", 10),
    ("epochs", 3),
    ("epochs", 5),
    ("epochs", 10)
]

PAIRES = {
    "films": [
        ("king", "queen"),
        ("war", "battle"),
        ("love", "romance"),
        ("murder", "killer")
    ],
    "amazon": [
        ("battery", "charge"),
        ("screen", "display"),
        ("camera", "picture"),
        ("charger", "cable")
    ]
}

random.seed(42)
resultats = []

for dataset, limite in [("films", 10000), ("amazon", 30000)]:

    print(f"\nChargement {dataset}...", flush=True)

    with open(RESULTS / f"{dataset}_tokens.json", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus = random.sample(corpus, min(limite, len(corpus)))

    for parametre, valeur in TESTS:

        config = {
            "vector_size": 100,
            "window": 5,
            "epochs": 5
        }
        config[parametre] = valeur

        print(
            f"{dataset} | {parametre}={valeur}",
            flush=True
        )

        debut = time.time()

        modele = Word2Vec(
            sentences=corpus,
            vector_size=config["vector_size"],
            window=config["window"],
            epochs=config["epochs"],
            min_count=3,
            workers=min(4, os.cpu_count() or 1),
            sg=0,
            seed=42
        )

        duree = round(time.time() - debut, 2)

        for mot1, mot2 in PAIRES[dataset]:
            if mot1 in modele.wv and mot2 in modele.wv:
                score = float(modele.wv.similarity(mot1, mot2))

                resultats.append({
                    "dataset": dataset,
                    "parametre": parametre,
                    "valeur": valeur,
                    "paire": f"{mot1}-{mot2}",
                    "similarite": round(score, 4),
                    "temps": duree
                })

        print(f"Termine en {duree}s", flush=True)

# Sauvegarde CSV
chemin = RESULTS / "hyperparameters_results.csv"

with open(chemin, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)

# Graphiques de comparaison
for dataset in ["films", "amazon"]:
    for parametre in ["vector_size", "window", "epochs"]:

        lignes = [
            r for r in resultats
            if r["dataset"] == dataset
            and r["parametre"] == parametre
        ]

        valeurs = sorted(set(r["valeur"] for r in lignes))

        fig, ax = plt.subplots(figsize=(10, 6))

        for paire in sorted(set(r["paire"] for r in lignes)):
            scores = []
            for valeur in valeurs:
                correspondances = [
                    r["similarite"] for r in lignes
                    if r["paire"] == paire
                    and r["valeur"] == valeur
                ]
                scores.append(
                    correspondances[0] if correspondances else float("nan")
                )

            ax.plot(valeurs, scores, marker="o", linewidth=2, label=paire)

        ax.set_title(
            f"Word2Vec - {dataset.upper()} - {parametre}",
            fontweight="bold"
        )
        ax.set_xlabel(parametre)
        ax.set_ylabel("Similarite cosinus")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            RESULTS / f"hyper_{dataset}_{parametre}.png",
            dpi=250
        )
        plt.close(fig)

print("\nEXPERIENCES TERMINEES !")
