#!/usr/bin/env python3
"""
simulador_CRNN — Validador de modelo CRNN con datos sísmicos reales

Desliza la ventana CRNN (2s) sobre toda la traza sísmica y genera un gráfico
con dos subplots: señales sísmicas (EW, NS, UD) y probabilidad CRNN vs tiempo.

Criterio de detección: 3 ventanas CONSECUTIVAS con CRNN > 0.9.

Uso:
  python3 simulador_CRNN.py DATA/EQ-2024/ISK0062401011618.csv
  python3 simulador_CRNN.py DATA/EQ-2023/
  python3 simulador_CRNN.py DATA/NonEQ/

© 2025 SkyAlert de México S.A. de C.V.
"""

import argparse
import glob
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CrowdQuake_RT import CrowdQuakeCRNN, FS, WINDOW_SAMPLES, INFERENCE_INTERVAL

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('simulador_CRNN')

GAL_TO_G = 1.0 / 980.665
SLIDE_SAMPLES = int(INFERENCE_INTERVAL * FS)  # 100 muestras = 1.0s
CONSEC_REQUIRED = 3  # Ventanas consecutivas requeridas para confirmar


def load_knet_csv(path):
    meta = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            line = line.strip()
            if ',' in line and line.startswith('#2'):
                parts = line[1:].split(',')
                if len(parts) >= 5:
                    meta['origin_time'] = parts[0]
                    meta['magnitude'] = float(parts[4])
            elif line.startswith('#') and ',' not in line:
                try:
                    val = int(line[1:])
                    if 'sampling_rate' not in meta:
                        meta['sampling_rate'] = val
                    else:
                        meta['duration_s'] = val
                except ValueError:
                    pass

    df = pd.read_csv(path, comment='#', header=None)
    ew = (df[3].values - df[3].mean()) * GAL_TO_G
    ns = (df[2].values - df[2].mean()) * GAL_TO_G
    ud = (df[4].values - df[4].mean()) * GAL_TO_G
    meta['pga_max_gal'] = max(np.max(np.abs(df[2].values - df[2].mean())),
                               np.max(np.abs(df[3].values - df[3].mean())),
                               np.max(np.abs(df[4].values - df[4].mean())))
    meta['type'] = 'EQ'
    return np.column_stack([ew, ns, ud]).astype(np.float32), meta


def load_noneq_csv(path):
    df = pd.read_csv(path, header=0)
    x = (df['x'].values - df['x'].mean()) / 9.80665
    y = (df['y'].values - df['y'].mean()) / 9.80665
    z = (df['z'].values - df['z'].mean()) / 9.80665
    meta = {
        'type': 'NonEQ',
        'category': os.path.basename(os.path.dirname(path)),
    }
    return np.column_stack([x, y, z]).astype(np.float32), meta


def load_file(path):
    with open(path, 'r') as f:
        first_line = f.readline().strip()
    if first_line == '#K-NET CSV':
        return load_knet_csv(path)
    elif 'dev_id' in first_line:
        return load_noneq_csv(path)
    else:
        raise ValueError(f"Formato no reconocido: {first_line}")


def has_consecutive(probs, threshold, n_required):
    """Retorna True si hay n_required ventanas consecutivas >= threshold."""
    streak = 0
    for p in probs:
        if p >= threshold:
            streak += 1
            if streak >= n_required:
                return True
        else:
            streak = 0
    return False


def validate_file(crnn, data, threshold):
    """Desliza ventana CRNN sobre toda la traza con batch prediction."""
    n = len(data)

    starts = list(range(0, n - WINDOW_SAMPLES + 1, SLIDE_SAMPLES))
    if not starts:
        return np.array([]), np.array([])

    batch = np.empty((len(starts), 2, 100, 3), dtype=np.float32)
    times = np.empty(len(starts), dtype=np.float64)

    for i, start in enumerate(starts):
        window = data[start:start + WINDOW_SAMPLES].astype(np.float32)
        window = window - window.mean(axis=0)
        batch[i] = window.reshape(2, 100, 3)
        times[i] = (start + WINDOW_SAMPLES / 2) / FS

    probs = crnn.model.predict(batch, batch_size=64, verbose=0).ravel()

    return times, probs


def plot_result(data, times, probs, meta, fname, threshold, confirmed, out_path):
    """Genera gráfico de 2 subplots: señales + probabilidad CRNN."""
    n = len(data)
    t_signal = np.arange(n) / FS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Título
    confirm_str = "CONFIRMADO ✅" if confirmed else "NO CONFIRMADO ❌"
    if meta['type'] == 'EQ':
        mag = f"M{meta.get('magnitude', '?')}" if 'magnitude' in meta else ""
        pga = f"PGA={meta.get('pga_max_gal', 0):.1f} gal"
        title = f"{fname}  —  SISMO {mag}  {pga}  [{confirm_str}]"
    else:
        title = f"{fname}  —  RUIDO ({meta.get('category', '?')})  [{confirm_str}]"
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Subplot 1: Señales sísmicas
    ax1.plot(t_signal, data[:, 0], lw=0.5, alpha=0.8, label='EW', color='#1f77b4')
    ax1.plot(t_signal, data[:, 1], lw=0.5, alpha=0.8, label='NS', color='#ff7f0e')
    ax1.plot(t_signal, data[:, 2], lw=0.5, alpha=0.8, label='UD', color='#2ca02c')
    ax1.set_ylabel('Aceleración (g)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', lw=0.3)

    # Subplot 2: Probabilidad CRNN
    ax2.plot(times, probs, lw=1.5, color='#d62728')
    ax2.axhline(threshold, color='k', ls='--', lw=1, alpha=0.6,
                label=f'Umbral={threshold} ({CONSEC_REQUIRED} consec.)')
    ax2.fill_between(times, probs, threshold, where=(probs >= threshold),
                     alpha=0.3, color='#d62728')
    ax2.set_ylabel('P(sismo)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Validador CRNN — Ventana deslizante sobre datos reales')
    parser.add_argument('input', help='Archivo CSV o carpeta')
    parser.add_argument('--model', default='result/CRNN_100Hz_10s.h5')
    parser.add_argument('--threshold', type=float, default=0.9)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)

    if not os.path.exists(model_path):
        log.error(f"Modelo no encontrado: {model_path}")
        sys.exit(1)

    # Recopilar archivos
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.csv')))
        if not files:
            files = sorted(glob.glob(os.path.join(args.input, '**', '*.csv'), recursive=True))
    else:
        files = [args.input]

    if not files:
        log.error(f"No se encontraron CSV en {args.input}")
        sys.exit(1)

    input_name = os.path.basename(os.path.normpath(args.input))
    if os.path.isfile(args.input):
        input_name = os.path.splitext(input_name)[0]
    out_dir = os.path.join(script_dir, 'result', 'REALTIME-RESULTS', input_name)
    os.makedirs(out_dir, exist_ok=True)

    crnn = CrowdQuakeCRNN(model_path)

    log.info("=" * 60)
    log.info(f"VALIDADOR CRNN — {len(files)} archivo(s)")
    log.info(f"Ventana: {WINDOW_SAMPLES/FS:.1f}s | Slide: {SLIDE_SAMPLES/FS:.1f}s")
    log.info(f"Confirmación: {CONSEC_REQUIRED} ventanas consecutivas > {args.threshold}")
    log.info(f"Salida: {out_dir}")
    log.info("=" * 60)

    eq_total = eq_detected = 0
    noneq_total = noneq_false = 0

    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        log.info(f"[{idx+1}/{len(files)}] {fname}")

        try:
            data, meta = load_file(fpath)
        except Exception as e:
            log.error(f"  Error: {e}")
            continue

        if len(data) < WINDOW_SAMPLES:
            log.warning(f"  Muy corto ({len(data)} muestras), saltando")
            continue

        t0 = time.perf_counter()
        times, probs = validate_file(crnn, data, args.threshold)
        dt = time.perf_counter() - t0

        max_prob = float(np.max(probs)) if len(probs) > 0 else 0.0
        n_above = int(np.sum(probs >= args.threshold))
        confirmed = has_consecutive(probs, args.threshold, CONSEC_REQUIRED)

        is_eq = meta['type'] == 'EQ'
        if is_eq:
            eq_total += 1
            if confirmed:
                eq_detected += 1
        else:
            noneq_total += 1
            if confirmed:
                noneq_false += 1

        status = "✅" if (confirmed == is_eq) else ("⚠️ FALSA ALARMA" if confirmed else "❌ NO DETECTADO")
        log.info(f"  {status} | max={max_prob:.4f} | >{args.threshold}: {n_above}/{len(probs)} | 3consec={'SÍ' if confirmed else 'NO'} | {dt:.1f}s")

        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.png')
        plot_result(data, times, probs, meta, fname, args.threshold, confirmed, out_path)

    # Resumen
    print("\n" + "=" * 60)
    print(f"RESUMEN — Umbral={args.threshold}, {CONSEC_REQUIRED} consecutivas")
    print("=" * 60)
    if eq_total > 0:
        print(f"  SISMOS:  {eq_detected}/{eq_total} confirmados → Recall = {eq_detected/eq_total*100:.1f}%")
    if noneq_total > 0:
        print(f"  RUIDO:   {noneq_false}/{noneq_total} falsas alarmas → Especificidad = {(noneq_total-noneq_false)/noneq_total*100:.1f}%")
    if eq_total + noneq_total > 0:
        total_correct = eq_detected + (noneq_total - noneq_false)
        total = eq_total + noneq_total
        print(f"  TOTAL:   {total_correct}/{total} correctos → Accuracy = {total_correct/total*100:.1f}%")
    print(f"  Gráficos: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
