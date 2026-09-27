"""
Arbeitspaket 5 (docs/Trainingskonzept_ML124.md, Abschnitt 7): Export eines Modells für Modul 124.

Erzeugt aus model/ml124/<name>/ (model.keras, norm.npz, config.json) zwei Header für SDS_110:

  ML124_Model_Data.hpp  Firmware: Schichtgrößen, Gewichte ([n_out][n_in], zeilenweise), Biases,
                        Aktivierung je Schicht, Mittelwert/Streuung je Eingang, Kontextlänge,
                        Merkmalsreihenfolge, Merkmalsversion, Modellname, ML_Test-Stand, Datum
  ML124_Model_Ref.hpp   nur Host-Test t_ml124: Referenzvektoren (Rohmerkmale, gestapelt wie in
                        124, und die Keras-Ausgaben)

Referenzvektoren: gleichmäßig verteilte Frames (ab SKIP_FRAMES) aus den Merkmalsdateien des
Testsplits (data/48kHz_features, split = test) oder aus den mit --ref angegebenen Dateien des
Merkmalswerkzeugs. Die Merkmalsversion der Dateien muss zur Version des Modells passen.

Prüfung nach dem Export: die Zahlen werden aus dem geschriebenen Header zurückgelesen und in
NumPy nachgerechnet; die maximale Abweichung zu Keras muss ≤ TOL sein, sonst Exit-Code 1.

Neuexport ohne neues Training (--same-as): Hat sich der SDS_110-Code nur außerhalb der Merkmale
geändert (z. B. Arraygeometrie in SDS_110_Config.hpp), ändert sich die Merkmalsversion, die Merkmale
aber nicht. Dann die Referenzdateien mit dem alten und dem neuen Merkmalswerkzeug erzeugen und
beide angeben: --ref <neu>/feat_*.npy --same-as <alt>. Der Export prüft, dass die alten Dateien die
Version des Modells tragen und die Merkmale gleichnamiger Dateien bitgleich sind, und schreibt die
neue Version (mit Vermerk im Header). Sonst Abbruch -> neu trainieren.

Aufruf (Repo-Wurzel, ~/ml_env):
  python app/train_ml124/export.py model/ml124/k5_h48_d3 [--out <verzeichnis>] [--n-ref 36] [--ref a.npy b.npy …]
                                   [--same-as <verzeichnis mit Merkmalsdateien der Modellversion>]
Ausgabe (Standard): model/ml124/<name>/export/
"""
import argparse, csv, json, os, re, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

from ml124_data import FEAT, NB, N_FEAT, SKIP_FRAMES, load_file, git_commit

TOL = 1e-5
ACT = {"relu": "Relu", "sigmoid": "Sigmoid"}
FEATURE_ORDER = "band_log_power[64], mel[40], spectral_flux, band_am_depth[64]"


def fmt(v):
    return "%.9g" % float(v)


def c_array(name, a, per_line=8):
    a = np.asarray(a, np.float32).ravel()
    lines = [", ".join(fmt(v) + "f" for v in a[i:i + per_line]) for i in range(0, len(a), per_line)]
    return f"constexpr float {name}[{len(a)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def dense_layers(model):
    out = []
    for l in model.layers:
        if type(l).__name__ != "Dense":
            continue
        W, b = l.get_weights()
        act = l.activation.__name__
        if act not in ACT:
            sys.exit(f"Aktivierung {act} in {l.name} wird von 124 nicht unterstützt")
        out.append((l.name, W.astype(np.float32), b.astype(np.float32), act))
    return out


def forward(x, mean, std, layers):
    """Referenz-Nachrechnung in float32, wie 124 (Normierung, Dense, Aktivierung)."""
    h = ((x - mean) / std).astype(np.float32)
    for W, b, act in layers:
        h = (h @ W + b).astype(np.float32)
        h = np.maximum(h, 0) if act == "relu" else (1.0 / (1.0 + np.exp(-h))).astype(np.float32)
    return h


def ref_files(args):
    if args.ref:
        return args.ref
    idx = os.path.join(FEAT, "index.csv")
    if not os.path.exists(idx):
        sys.exit(f"{idx} fehlt – Merkmalsdateien mit --ref angeben")
    return [os.path.join(FEAT, r["file"]) for r in csv.DictReader(open(idx)) if r["split"] == "test"]


def pick_refs(files, context, n, fver):
    """n Frames, gleichmäßig über die Dateien und innerhalb jeder Datei verteilt."""
    per = max(1, int(np.ceil(n / len(files))))
    step = max(1, len(files) // n) if len(files) > n else 1
    xs, src = [], []
    for f in files[::step]:
        d = load_file(f, context)
        if d["feature_version"] != fver:
            sys.exit(f"{f}: Merkmalsversion {d['feature_version']}, Modell {fver}")
        rows = np.linspace(0, len(d["x"]) - 1, per).astype(int)
        xs.append(d["x"][rows]); src += [f"{os.path.basename(f)}:{SKIP_FRAMES + r}" for r in rows]
    x = np.concatenate(xs)[:n].astype(np.float32)
    return x, src[:n]


def same_features(new_files, old_dir, model_ver):
    """Neuexport: neue Merkmalsdateien bitgleich zu den alten (Version des Modells)? -> neue Version."""
    new_ver = None
    for f in new_files:
        g = os.path.join(old_dir, os.path.basename(f))
        if not os.path.exists(g):
            sys.exit(f"--same-as: {g} fehlt")
        jn, jo = (json.load(open(p[:-4] + ".json")) for p in (f, g))
        if jo["feature_version"] != model_ver:
            sys.exit(f"{g}: Merkmalsversion {jo['feature_version']}, Modell {model_ver}")
        if new_ver not in (None, jn["feature_version"]):
            sys.exit(f"{f}: uneinheitliche Merkmalsversion {jn['feature_version']} / {new_ver}")
        new_ver = jn["feature_version"]
        if not np.array_equal(np.load(f), np.load(g)):
            sys.exit(f"{f}: Merkmale weichen von {g} ab -> Modell neu trainieren")
    print(f"Merkmale {model_ver} -> {new_ver} bitgleich ({len(new_files)} Dateien)")
    return new_ver, (f"// Merkmalsversion {new_ver}: Modell trainiert mit {model_ver}, Merkmale bitgleich "
                     f"(export.py --same-as, {len(new_files)} Referenzdateien)\n")


def read_back(path, name):
    """Array `name` aus einem geschriebenen Header lesen (prüft Formatierung und Rundung)."""
    txt = open(path).read()
    m = re.search(rf"constexpr float {name}\[\d+\] = \{{(.*?)\}};", txt, re.S)
    return np.array([float(v.rstrip("f")) for v in re.findall(r"[-+0-9.eE]+f", m.group(1))], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-ref", type=int, default=36)
    ap.add_argument("--ref", nargs="*", default=None, help="Merkmalsdateien (.npy) statt des Testsplits")
    ap.add_argument("--same-as", default=None, help="Merkmalsdateien der Modellversion (Neuexport, siehe oben)")
    a = ap.parse_args()

    import tensorflow as tf
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    norm = np.load(os.path.join(a.model_dir, "norm.npz"))
    mean, std = norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
    model = tf.keras.models.load_model(os.path.join(a.model_dir, "model.keras"), compile=False)
    layers = dense_layers(model)
    K, n_in = cfg["context"], cfg["n_inputs"]
    assert n_in == K * N_FEAT == len(mean) == layers[0][1].shape[0], (n_in, K, len(mean))
    assert layers[-1][1].shape[1] == NB

    fver, note = cfg["feature_version"], ""
    if a.same_as:
        fver, note = same_features(ref_files(a), a.same_as, cfg["feature_version"])
    x_ref, src = pick_refs(ref_files(a), K, a.n_ref, fver)
    y_ref = model.predict((x_ref - mean) / std, verbose=0).astype(np.float32)

    out = a.out or os.path.join(a.model_dir, "export"); os.makedirs(out, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    head = (f"// AUTO-GENERATED by ML_Test/app/train_ml124/export.py – nicht von Hand ändern\n"
            f"// Modell {cfg['name']} (ML_Test {cfg['ml_test_git']}, trainiert {cfg['date']}), "
            f"Export {stamp} (ML_Test {git_commit()})\n" + note)
    widths = [W.shape[1] for _, W, _, _ in layers]

    # --- Firmware-Header ------------------------------------------------------------------
    p_data = os.path.join(out, "ML124_Model_Data.hpp")
    with open(p_data, "w") as fh:
        fh.write(head + "// Modul 124: MLP für s(t) = (p_1 .. p_B), Docs: ML_Test docs/Training_ML124.md\n"
                 "#pragma once\n#include <cstdint>\n\nnamespace sds110::ml124model {\n\n")
        fh.write(f'constexpr char     MODEL_FEATURE_VERSION[] = "{fver}";\n'
                 f'constexpr char     MODEL_NAME[]            = "{cfg["name"]}";\n'
                 f'constexpr char     ML_TEST_GIT[]           = "{cfg["ml_test_git"]}";\n'
                 f"constexpr uint32_t NUM_FEATURES = {N_FEAT};   // je Frame: {FEATURE_ORDER}\n"
                 f"constexpr uint32_t CONTEXT      = {K};     // Frames t, t−1, …, t−{K - 1} (neuester zuerst); "
                 f"am Anfang mit dem ersten Frame aufgefüllt\n"
                 f"constexpr uint32_t NUM_INPUTS   = {n_in};\n"
                 f"constexpr uint32_t NUM_OUTPUTS  = {NB};\n"
                 f"constexpr uint32_t NUM_LAYERS   = {len(layers)};\n"
                 f"constexpr uint32_t MAX_WIDTH    = {max(widths)};    // breiteste Schicht nach dem Eingang\n"
                 f"constexpr uint32_t NUM_PARAMS   = {cfg['params']};\n\n"
                 "enum class Act : uint8_t { Relu, Sigmoid };\n"
                 "struct Layer { const float* W; const float* b; uint32_t n_in; uint32_t n_out; Act act; };\n\n"
                 "// Normierung: x_n = (x − MEAN) / STD\n")
        fh.write(c_array("MEAN", mean) + "\n" + c_array("STD", std) + "\n")
        for i, (name, W, b, act) in enumerate(layers):
            fh.write(f"// Schicht {i} ({name}): {W.shape[0]} -> {W.shape[1]}, {act}; W[n_out][n_in]\n")
            fh.write(c_array(f"W{i}", W.T) + "\n" + c_array(f"B{i}", b) + "\n")
        fh.write("constexpr Layer LAYERS[NUM_LAYERS] = {\n" + ",\n".join(
            f"    {{ W{i}, B{i}, {W.shape[0]}, {W.shape[1]}, Act::{ACT[act]} }}" for i, (_, W, _, act) in enumerate(layers))
            + "\n};\n\n} // namespace sds110::ml124model\n")

    # --- Referenzvektoren (Host-Test) -----------------------------------------------------
    p_ref = os.path.join(out, "ML124_Model_Ref.hpp")
    with open(p_ref, "w") as fh:
        fh.write(head + "// Referenzvektoren für test/host/t_ml124 (nicht in die Firmware einbinden)\n"
                 "// Eingänge: Rohmerkmale, gestapelt wie in 124; Ausgänge: Keras model.predict\n"
                 "// Herkunft (Datei:Frame):\n" + "".join(f"//   {s}\n" for s in src) +
                 "#pragma once\n#include <cstdint>\n\nnamespace sds110::ml124model {\n\n"
                 f"constexpr uint32_t REF_N = {len(x_ref)};\n")
        fh.write(c_array("REF_X", x_ref) + "\n" + c_array("REF_Y", y_ref) + "\n} // namespace sds110::ml124model\n")

    # --- Prüfung: Header zurücklesen und nachrechnen ---------------------------------------
    rb = [(read_back(p_data, f"W{i}").reshape(W.shape[1], W.shape[0]).T, read_back(p_data, f"B{i}"), act)
          for i, (_, W, _, act) in enumerate(layers)]
    y_np = forward(read_back(p_ref, "REF_X").reshape(x_ref.shape), read_back(p_data, "MEAN"), read_back(p_data, "STD"), rb)
    err = float(np.max(np.abs(y_np - read_back(p_ref, "REF_Y").reshape(y_ref.shape))))
    kb = (sum(W.size + b.size for _, W, b, _ in layers) + 2 * n_in) * 4 / 1024
    print(f"{p_data}\n{p_ref}\n{len(layers)} Schichten {n_in}->{'->'.join(map(str, widths))}, "
          f"{kb:.1f} kB float32, {len(x_ref)} Referenzvektoren, max |NumPy − Keras| = {err:.2e} (Grenze {TOL:g})")
    if err > TOL:
        sys.exit(1)


if __name__ == "__main__":
    main()
