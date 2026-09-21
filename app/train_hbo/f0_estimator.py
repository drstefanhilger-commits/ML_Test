import numpy as np

def estimate_f0_v3(S, freqs, rotor_min=80.0, rotor_max=200.0,
                   n_harmonics=6, freq_tol=5.0):

    spec_mean = np.mean(S, axis=1)

    rotor_mask = (freqs >= rotor_min) & (freqs <= rotor_max)
    rotor_freqs = freqs[rotor_mask]
    rotor_spec = spec_mean[rotor_mask]

    # Alte Peak-Erkennung: einfache lokale Maxima
    candidates = []
    for i in range(1, len(rotor_spec) - 1):
        if rotor_spec[i] > rotor_spec[i - 1] and rotor_spec[i] > rotor_spec[i + 1]:
            candidates.append(i)

    if not candidates:
        return rotor_freqs[np.argmax(rotor_spec)]

    best_f0 = None
    best_score = -np.inf

    for ci in candidates:
        f0 = rotor_freqs[ci]

        harmonics = [f0 * n for n in range(1, n_harmonics + 1)]

        energies = []
        for h in harmonics:
            idx = np.argmin(np.abs(freqs - h))
            # Alte Toleranzprüfung
            if abs(freqs[idx] - h) > freq_tol:
                energies.append(0.0)
            else:
                energies.append(spec_mean[idx])

        energies = np.array(energies)

        # Alte Version: mindestens 3 Harmonische > 0
        nonzero = np.sum(energies > 0.0)
        if nonzero < 3:
            continue

        # Alte Energieabfall-Bewertung
        decay_score = 0.0
        for i in range(len(energies) - 1):
            if energies[i] > energies[i + 1]:
                decay_score += 1.0

        harmonic_energy = np.sum(energies)

        # Alter Score ohne 10%-Schwelle
        score = nonzero + decay_score + np.log(harmonic_energy + 1e-9)

        if score > best_score:
            best_score = score
            best_f0 = f0

    if best_f0 is None:
        return rotor_freqs[np.argmax(rotor_spec)]

    return best_f0
