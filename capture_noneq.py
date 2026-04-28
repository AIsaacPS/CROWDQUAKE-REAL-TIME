#!/usr/bin/env python3
"""
capture_noneq.py — Captura datos de ruido ambiental del AnyShake para entrenamiento

Graba las 3 componentes del AnyShake y las guarda en CSVs compatibles con
el pipeline de entrenamiento CRNN (mismo formato que NonEQ de CrowdQuake).

Genera archivos de 5 minutos (30,000 muestras @ 100 Hz) en:
  DATA/NonEQ/knuEqAccData_AnyShake_<categoría>/DeviceXX.csv

Uso:
  python3 capture_noneq.py --category Quiet --duration 60
  python3 capture_noneq.py --category Day --duration 120
  python3 capture_noneq.py --category Traffic --duration 60

© 2025 SkyAlert de México S.A. de C.V.
"""

import argparse
import os
import socket
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

# AnyShake
HOST = 'localhost'
PORT = 30000
COUNT_TO_MS2 = 0.059 * 0.01  # counts → gal → m/s²
FILE_DURATION = 300  # 5 minutos por archivo
FS = 100


def connect(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((host, port))
    sock.sendall(b"AT+REALTIME=1\r\n")
    time.sleep(1)
    sock.setblocking(False)
    return sock


def capture(host, port, duration_min, category, out_dir):
    sock = connect(host, port)
    print(f"Conectado a AnyShake {host}:{port}")
    print(f"Categoría: {category}")
    print(f"Duración: {duration_min} minutos")
    print(f"Salida: {out_dir}")
    print(f"Archivos de {FILE_DURATION}s cada uno")
    print("=" * 50)
    print("Capturando... (Ctrl+C para detener)")

    comp = {'ENZ': deque(), 'ENE': deque(), 'ENN': deque()}
    text_buf = ""

    samples = []  # (ew, ns, ud) en m/s²
    file_count = 0
    total_samples = 0
    target_samples = int(duration_min * 60 * FS)
    file_samples = FILE_DURATION * FS
    start_time = time.time()

    os.makedirs(out_dir, exist_ok=True)

    try:
        while total_samples < target_samples:
            # Leer TCP
            raw = b""
            try:
                while True:
                    chunk = sock.recv(4096)
                    if chunk:
                        raw += chunk
                    else:
                        break
            except BlockingIOError:
                pass
            except Exception:
                pass

            if not raw:
                time.sleep(0.005)
                continue

            text_buf += raw.decode('utf-8', errors='ignore')
            if len(text_buf) > 16384:
                text_buf = text_buf[-8192:]

            lines = text_buf.split('\r')
            text_buf = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line.startswith('$') or '*' not in line:
                    continue
                main_part = line.split('*', 1)[0]
                parts = main_part.split(',')
                if len(parts) < 8:
                    continue
                c = parts[4]
                if c not in comp:
                    continue
                for p in parts[7:]:
                    p = p.strip()
                    if p and not p.startswith('*'):
                        try:
                            comp[c].append(int(p))
                        except ValueError:
                            pass

            # Sincronizar componentes
            while all(len(comp[c]) > 0 for c in ('ENZ', 'ENE', 'ENN')):
                enz = comp['ENZ'].popleft()
                ene = comp['ENE'].popleft()
                enn = comp['ENN'].popleft()
                # x=EW(ENE), y=NS(ENN), z=UD(ENZ) en m/s²
                samples.append((ene * COUNT_TO_MS2, enn * COUNT_TO_MS2, enz * COUNT_TO_MS2))
                total_samples += 1

                # Guardar archivo cada FILE_DURATION segundos
                if len(samples) >= file_samples:
                    _save_file(samples[:file_samples], file_count, out_dir)
                    file_count += 1
                    samples = samples[file_samples:]
                    elapsed = time.time() - start_time
                    remaining = (duration_min * 60) - elapsed
                    print(f"  Archivo {file_count:02d} guardado | "
                          f"{total_samples/FS:.0f}s capturados | "
                          f"~{remaining/60:.1f} min restantes")

    except KeyboardInterrupt:
        print("\nDetenido por usuario")
    finally:
        sock.close()

    # Guardar residuo si tiene al menos 30s
    if len(samples) >= 30 * FS:
        _save_file(samples, file_count, out_dir)
        file_count += 1
        print(f"  Archivo {file_count:02d} guardado (parcial, {len(samples)/FS:.0f}s)")

    print("=" * 50)
    print(f"Total: {file_count} archivos, {total_samples/FS:.0f}s, "
          f"{total_samples} muestras")
    print(f"Guardados en: {out_dir}")


def _save_file(samples, idx, out_dir):
    """Guarda muestras en formato NonEQ compatible."""
    arr = np.array(samples)
    ts_start = int(time.time() * 1000)
    ts = ts_start + np.arange(len(arr)) * 10  # 10ms entre muestras @ 100Hz

    df = pd.DataFrame({
        'dev_id': f'{idx:02d}',
        'x': arr[:, 0],
        'y': arr[:, 1],
        'z': arr[:, 2],
        'ts': ts.astype(np.int64),
    })
    fname = os.path.join(out_dir, f'Device{idx:02d}.csv')
    df.to_csv(fname, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Captura datos de ruido ambiental del AnyShake para entrenamiento CRNN')
    parser.add_argument('--category', required=True,
                        help='Nombre de la categoría (ej: Quiet, Day, Traffic)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duración en minutos (default: 60)')
    parser.add_argument('--host', default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'CSV-Data', 'NonEQ',
                           f'knuEqAccData_AnyShake_{args.category}')

    capture(args.host, args.port, args.duration, args.category, out_dir)


if __name__ == '__main__':
    main()
