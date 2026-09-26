"""
Arbeitspaket 4 (docs/Trainingskonzept_ML124.md, Abschnitt 6): Training des MLP für Modul 124.

Architektur: 169·K → 128 (ReLU) → 64 (ReLU) → 64 (Sigmoid), K = Kontext in Frames (Standard;
verdeckte Schichten mit --hidden wählbar, Flash-Grenze 200 kB = 51 200 float32-Parameter).
Normierung: Mittelwert/Streuung je Eingang aus den Trainingsframes (im Modellordner abgelegt).
Verlust: binäre Kreuzentropie je Band mit weichen Zielen y_b = σ(SNR_b / 3 dB); keine
Gewichtung, da der Anteil positiver Bänder ca. 50 % beträgt (docs/Datensatz_ML124.md).
Validierung: 15 % der Drohnen-Clips (alle ihre Gemische) und 15 % der Umwelt-Beispiele des
Trainingssplits; der Testsplit (Fold 5) wird hier nicht angefasst.

⚠ PRÜFEN: Die Verwendung der Drohnendaten im Training ist noch nicht freigegeben
(docs/Daten_ML124.md). Das Modell erbt diesen Vorbehalt.

Aufruf (Repo-Wurzel, ~/ml_env):
  python app/train_ml124/train.py [--context 1] [--dropout 0.3] [--l2 1e-4] [--name …] [--exclude-rotors 6]
Ausgabe: model/ml124/<name>/ model.keras, norm.npz, config.json, history.csv
"""
import argparse, csv, json, os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from ml124_data import ROOT, NB, N_FEAT, SKIP_FRAMES, LABEL_SCALE_DB, load_dataset, expand, soft_target, git_commit

SEED = 20260928
HIDDEN = (128, 64)
VAL_FRACTION = 0.15
BATCH = 512
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 10


def build_model(n_in, hidden=HIDDEN, dropout=0.0, l2=0.0):
    """Dropout und L2 wirken nur im Training; die Inferenz (Export, 124) sieht reine Dense-Schichten."""
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    x = inp = tf.keras.Input(shape=(n_in,), name="features_norm")
    if dropout:
        x = tf.keras.layers.Dropout(dropout / 2, name="dropout_in")(x)
    for i, h in enumerate(hidden):
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg, name=f"dense_{i}")(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    out = tf.keras.layers.Dense(NB, activation="sigmoid", kernel_regularizer=reg, name="s_t")(x)
    return tf.keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout nach den verdeckten Schichten (Eingang: halber Wert)")
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)), help="verdeckte Schichten, z. B. 64,64")
    ap.add_argument("--exclude-rotors", type=int, default=0,
                    help="Hold-out nach Drohnentyp: Drohnen mit dieser Rotorzahl nicht trainieren")
    a = ap.parse_args()
    name = a.name or f"ml124_k{a.context}" + (f"_ohne{a.exclude_rotors}rot" if a.exclude_rotors else "")
    out = os.path.join(ROOT, "model", "ml124", name); os.makedirs(out, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED); tf.config.experimental.enable_op_determinism()

    t0 = time.time()
    x, snr, _, info, fver = load_dataset("train", a.context)
    rng = np.random.default_rng(SEED)
    drones = sorted({r["drone_file"] for r in info if r["drone_file"]})
    val_drones = set(rng.choice(drones, int(round(len(drones) * VAL_FRACTION)), replace=False))
    noise_ids = [r["id"] for r in info if r["noise_only"]]
    val_noise = set(rng.choice(noise_ids, int(round(len(noise_ids) * VAL_FRACTION)), replace=False))
    ex_val = np.array([r["drone_file"] in val_drones or r["id"] in val_noise for r in info])
    ex_skip = np.array([bool(a.exclude_rotors) and r["drone_type"].startswith(f"{a.exclude_rotors}x") for r in info])
    fr_val = np.repeat(ex_val, [r["n"] for r in info]); fr_skip = np.repeat(ex_skip, [r["n"] for r in info])
    tr, va = ~fr_val & ~fr_skip, fr_val & ~fr_skip

    mean = x[tr].mean(0); std = x[tr].std(0); std[std < 1e-6] = 1.0
    xn = (x - mean) / std; y = soft_target(snr)
    print(f"Daten: {len(x)} Frames ({tr.sum()} Training, {va.sum()} Validierung, {fr_skip.sum()} ausgeschlossen), "
          f"{x.shape[1]} Eingänge, geladen in {time.time() - t0:.0f} s")

    hidden = tuple(int(v) for v in a.hidden.split(","))
    model = build_model(x.shape[1], hidden, a.dropout, a.l2)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy")
    cb = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
          tf.keras.callbacks.CSVLogger(os.path.join(out, "history.csv"))]
    t0 = time.time()
    h = model.fit(xn[tr], y[tr], validation_data=(xn[va], y[va]), batch_size=BATCH, epochs=MAX_EPOCHS,
                  shuffle=True, callbacks=cb, verbose=2)
    best = int(np.argmin(h.history["val_loss"]))
    model.save(os.path.join(out, "model.keras"))
    np.savez(os.path.join(out, "norm.npz"), mean=mean.astype(np.float32), std=std.astype(np.float32))
    cfg = dict(name=name, context=a.context, n_features=N_FEAT, n_inputs=int(x.shape[1]), hidden=list(hidden),
               activations=["relu"] * len(hidden) + ["sigmoid"], n_bands=NB, label="sigmoid(snr_db/%g)" % LABEL_SCALE_DB,
               skip_frames=SKIP_FRAMES, dropout=a.dropout, l2=a.l2, seed=SEED, batch=BATCH, lr=LR, epochs_run=len(h.history["loss"]),
               best_epoch=best + 1, train_loss=float(h.history["loss"][best]), val_loss=float(h.history["val_loss"][best]),
               exclude_rotors=a.exclude_rotors, val_drones=sorted(val_drones), frames_train=int(tr.sum()),
               frames_val=int(va.sum()), feature_version=fver, ml_test_git=git_commit(),
               tensorflow=tf.__version__, date=time.strftime("%Y-%m-%d"), train_time_s=round(time.time() - t0, 1),
               params=int(model.count_params()))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"{name}: beste Epoche {best + 1}, val_loss {cfg['val_loss']:.4f}, {cfg['params']} Parameter, "
          f"{cfg['train_time_s']} s -> {out}")


if __name__ == "__main__":
    main()
