"""Evaluate modelo.h5 on Test/ without retraining.

Usage: python evaluate.py <repo_dir> <out_dir>

Reports three views, because the dataset has many byte-identical images:
  all          every file in Test/
  unique       one copy of each distinct image in Test/
  unseen       unique images whose exact bytes do not appear in Train/
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

CLASES = ["Primera", "Segunda", "Tercera"]  # alphabetical: flow_from_directory order
NOMBRES = ["Motor-vehicle lane", "No parking", "No entry"]
IMG = 160
UMBRAL = 0.85


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def cargar(path):
    # flow_from_directory uses nearest-neighbour resize and rescale=1./255
    img = Image.open(path).convert("RGB").resize((IMG, IMG), Image.NEAREST)
    return np.asarray(img, dtype="float32") / 255.0


def resumen(y, pred, conf):
    aceptadas = conf >= UMBRAL
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(
            y, pred, labels=[0, 1, 2], target_names=NOMBRES, output_dict=True, zero_division=0
        ),
        "threshold_0.85": {
            "coverage": round(float(aceptadas.mean()), 4),
            "accuracy_when_accepted": round(float(accuracy_score(y[aceptadas], pred[aceptadas])), 4)
            if aceptadas.any()
            else None,
        },
    }


def main():
    repo, salida = Path(sys.argv[1]), Path(sys.argv[2])
    salida.mkdir(parents=True, exist_ok=True)
    modelo = load_model(repo / "modelo.h5", compile=False)

    hashes_train = {md5(p) for p in (repo / "Train").rglob("*.png")}

    rutas, y, hashes = [], [], []
    for i, clase in enumerate(CLASES):
        for p in sorted((repo / "Test" / clase).iterdir()):
            rutas.append(p)
            y.append(i)
            hashes.append(md5(p))
    y = np.array(y)

    x = np.stack([cargar(p) for p in rutas])
    probs = modelo.predict(x, batch_size=32, verbose=0)
    pred, conf = probs.argmax(1), probs.max(1)

    vistos, unica = set(), []
    for h in hashes:
        unica.append(h not in vistos)
        vistos.add(h)
    unica = np.array(unica)
    no_vista = unica & np.array([h not in hashes_train for h in hashes])

    res = {
        "all": resumen(y, pred, conf),
        "unique": resumen(y[unica], pred[unica], conf[unica]),
        "unseen": resumen(y[no_vista], pred[no_vista], conf[no_vista]),
        "params": int(modelo.count_params()),
    }
    (salida / "results.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    for vista in ("all", "unique", "unseen"):
        r = res[vista]
        print(vista, "n=", r["n"], "acc=", r["accuracy"], "cm=", r["confusion_matrix"], r["threshold_0.85"])
    for i in np.where(no_vista & (pred != y))[0]:
        print("  error:", rutas[i].name, NOMBRES[y[i]], "->", NOMBRES[pred[i]], round(float(conf[i]), 3))

    # Figure for the README: confusion matrix on the strictest view
    cm = np.array(res["unseen"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), NOMBRES, rotation=20, ha="right")
    ax.set_yticks(range(3), NOMBRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Unique test images not in Train (n={res['unseen']['n']})")
    for (f, c), v in np.ndenumerate(cm):
        ax.text(c, f, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(salida / "confusion_matrix.png")

    # Figure for the README: sample predictions
    rng = np.random.default_rng(7)
    indices = rng.choice(np.where(no_vista)[0], size=min(8, int(no_vista.sum())), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.4), dpi=120)
    for ax, i in zip(axes.flat, indices):
        ax.imshow(x[i])
        ax.axis("off")
        ok = pred[i] == y[i]
        ax.set_title(f"{NOMBRES[pred[i]]}\n{conf[i]:.0%}", fontsize=9, color="#1a7f37" if ok else "#cf222e")
    for ax in axes.flat[len(indices):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(salida / "sample_predictions.png")


if __name__ == "__main__":
    main()
