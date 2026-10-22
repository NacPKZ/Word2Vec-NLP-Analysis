
import json
import os
import time
import csv
from pathlib import Path
from gensim.models import Word2Vec

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# Parametres communs
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 5
EPOCHS = 5
WORKERS = min(4, os.cpu_count() or 1)

def entrainer(nom, architecture):
    sg = 0 if architecture == "cbow" else 1

    print(f"\nChargement : {nom}", flush=True)

    with open(RESULTS / f"{nom}_tokens.json", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Documents : {len(corpus)}", flush=True)
    print(f"Entrainement : {nom} / {architecture}", flush=True)

    debut = time.time()

    modele = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=sg,
        epochs=EPOCHS,
        seed=42
    )

    duree = round(time.time() - debut, 2)

    chemin = MODELS / f"{nom}_{architecture}.model"
    modele.save(str(chemin))

    vocabulaire = len(modele.wv)

    print("Temps :", duree, "secondes", flush=True)
    print("Vocabulaire :", vocabulaire, "mots", flush=True)
    print("Modele sauvegarde :", chemin.name, flush=True)

    return {
        "dataset": nom,
        "architecture": architecture,
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "epochs": EPOCHS,
        "vocabulaire": vocabulaire,
        "temps_secondes": duree
    }

resultats = []

for nom in ["films", "amazon"]:
    for architecture in ["cbow", "skipgram"]:
        resultat = entrainer(nom, architecture)
        resultats.append(resultat)

with open(
    RESULTS / "training_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)

print("\nLES 4 MODELES SONT ENTRAINES !")
