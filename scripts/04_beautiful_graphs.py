
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from adjustText import adjust_text
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

GROUPS = {
    "films": ["king", "love", "war", "family", "murder"],
    "amazon": ["phone", "battery", "screen", "camera", "charger", "case"]
}

COLORS = [
    "#2563EB", "#E11D48", "#059669",
    "#D97706", "#7C3AED", "#0891B2"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 19,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFD",
    "axes.edgecolor": "#CBD5E1",
    "grid.color": "#E2E8F0",
    "savefig.facecolor": "white"
})

for dataset, mots in GROUPS.items():
    for architecture in ["cbow", "skipgram"]:
        nom = f"{dataset}_{architecture}"
        print(f"Creation graphique : {nom}", flush=True)

        model = Word2Vec.load(str(MODELS / f"{nom}.model"))
        wv = model.wv

        groups = {}
        for mot in mots:
            if mot in wv:
                groups[mot] = [
                    voisin for voisin, _ in
                    wv.most_similar(mot, topn=3)
                ]

        # Chaque mot unique est projete une seule fois
        selection = list(dict.fromkeys(
            m for centre, voisins in groups.items()
            for m in [centre] + voisins
        ))

        vectors = np.array([wv[m] for m in selection])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        positions = dict(zip(selection, coords))

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_axisbelow(True)
        ax.grid(alpha=0.7, linestyle="--")

        texts = []
        legend = []

        for index, (centre, voisins) in enumerate(groups.items()):
            color = COLORS[index % len(COLORS)]
            cx, cy = positions[centre]

            # Mot central
            ax.scatter(
                cx, cy, s=220, c=color,
                edgecolors="white", linewidths=2,
                zorder=5
            )

            t = ax.text(
                cx, cy, centre.upper(),
                fontsize=12, fontweight="bold",
                color=color, zorder=6
            )
            texts.append(t)

            for voisin in voisins:
                vx, vy = positions[voisin]

                ax.plot(
                    [cx, vx], [cy, vy],
                    color=color, alpha=0.25,
                    linewidth=1.5, zorder=1
                )

                ax.scatter(
                    vx, vy, s=75, c=color,
                    alpha=0.8, edgecolors="white",
                    linewidths=1, zorder=3
                )

                t = ax.text(
                    vx, vy, voisin,
                    fontsize=10, color="#334155",
                    zorder=6
                )
                texts.append(t)

            legend.append(
                Line2D(
                    [0], [0], marker="o",
                    color="none", label=centre.upper(),
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=10
                )
            )

        adjust_text(
            texts,
            ax=ax,
            expand=(1.3, 1.5),
            force_text=(0.7, 0.9),
            arrowprops=dict(
                arrowstyle="-",
                color="#94A3B8",
                lw=0.7,
                alpha=0.7
            )
        )

        for t in texts:
            t.set_path_effects([
                patheffects.withStroke(
                    linewidth=3,
                    foreground="white"
                )
            ])

        titre_dataset = (
            "RESUMES DE FILMS"
            if dataset == "films"
            else "AVIS CLIENTS AMAZON"
        )

        ax.set_title(
            f"WORD2VEC | {titre_dataset}\n"
            f"Architecture : {architecture.upper()}",
            fontweight="bold", pad=25
        )

        ax.set_xlabel("Composante principale 1")
        ax.set_ylabel("Composante principale 2")

        ax.legend(
            handles=legend,
            title="Groupes semantiques",
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=False
        )

        fig.text(
            0.5, 0.025,
            "Projection PCA 2D des vecteurs Word2Vec "
            "| Les distances sont indicatives",
            ha="center", fontsize=10,
            color="#64748B"
        )

        fig.subplots_adjust(
            left=0.08, right=0.78,
            top=0.85, bottom=0.12
        )

        chemin = RESULTS / f"beautiful_{nom}.png"
        fig.savefig(chemin, dpi=300)
        plt.close(fig)

        print(f"Enregistre : {chemin.name}")

print("\n4 GRAPHIQUES AMELIORES TERMINES !")
