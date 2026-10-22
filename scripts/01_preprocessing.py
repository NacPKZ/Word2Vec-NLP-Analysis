
import pandas as pd
import json
import gzip
from pathlib import Path
from gensim.utils import simple_preprocess

# Localiser automatiquement le dossier du projet
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

def nettoyer(texte):
    return simple_preprocess(
        str(texte),
        deacc=True,
        min_len=2
    )

# 1. FILMS
print("Chargement des films...", flush=True)

films = pd.read_csv(
    DATA / "movies_metadata.csv",
    usecols=["overview"],
    low_memory=False
)

tokens_films = [
    nettoyer(t)
    for t in films["overview"].dropna()
]

tokens_films = [t for t in tokens_films if len(t) >= 3]
print("Films nettoyes :", len(tokens_films), flush=True)

# 2. AMAZON
print("Chargement Amazon...", flush=True)

fichiers = [
    DATA / "Cell_Phones_and_Accessories_5.json",
    DATA / "Cell_Phones_and_Accessories_5.json.gz",
    DATA / "reviews_Cell_Phones_and_Accessories_5.json.gz"
]

fichier_amazon = next((p for p in fichiers if p.exists()), None)

if fichier_amazon is None:
    raise FileNotFoundError("Fichier Amazon introuvable dans data")

# Detecter le format reel du fichier
with open(fichier_amazon, "rb") as f:
    compresse = f.read(2) == b"\x1f\x8b"

ouvrir = gzip.open if compresse else open

tokens_amazon = []

with ouvrir(fichier_amazon, "rt", encoding="utf-8") as f:
    for i, ligne in enumerate(f, 1):
        if not ligne.strip():
            continue

        avis = json.loads(ligne).get("reviewText", "")
        tokens = nettoyer(avis)

        if len(tokens) >= 3:
            tokens_amazon.append(tokens)

        if i % 100000 == 0:
            print(f"{i} avis lus...", flush=True)

print("Avis nettoyes :", len(tokens_amazon), flush=True)

# 3. SAUVEGARDE
for nom, corpus in [
    ("films", tokens_films),
    ("amazon", tokens_amazon)
]:
    chemin = RESULTS / f"{nom}_tokens.json"

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"{nom} : {len(corpus)} textes")
    if corpus:
        print("Exemple :", corpus[0][:15])

print("\nPRETRAITEMENT TERMINE", flush=True)
