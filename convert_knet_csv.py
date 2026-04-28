#!/usr/bin/env python3
"""
Convierte archivos K-NET CSV (DATA/EQ-*/) al formato .txt que esperan CRNN.py y ANN.py.

Entrada:  K-NET CSV con 16 líneas de header (#), columnas Time,RelTime,N-S(gal),E-W(gal),U-D(gal)
Salida:   .txt con 3 columnas separadas por espacio: EW  NS  UD (en g, media restada)

El orden EW, NS, UD corresponde a X, Y, Z en CRNN.py (Processing.py original usa el mismo orden).

Uso:
    python convert_knet_csv.py                    # PGA threshold 0.05g (default)
    python convert_knet_csv.py --threshold 0.1    # PGA threshold 0.1g
    python convert_knet_csv.py --no-filter         # Sin filtro de PGA, convierte todo
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'CSV-Data')
OUTPUT_DIR = os.path.join(BASE, 'EQ')
GAL_TO_G = 1.0 / 980.665


def convert_file(fn, threshold):
    """Convierte un archivo K-NET CSV a .txt. Retorna True si se guardó."""
    df = pd.read_csv(fn, comment='#', header=None)
    # Columnas: 0=Time, 1=RelativeTime, 2=N-S(gal), 3=E-W(gal), 4=U-D(gal)
    ew = df[3].values
    ns = df[2].values
    ud = df[4].values

    # Convertir gal → g y restar media (offset)
    ew_g = (ew - np.mean(ew)) * GAL_TO_G
    ns_g = (ns - np.mean(ns)) * GAL_TO_G
    ud_g = (ud - np.mean(ud)) * GAL_TO_G

    pga = np.max(ew_g)

    if threshold is not None and pga < threshold:
        return False, pga

    name = os.path.splitext(os.path.basename(fn))[0]
    np.savetxt(os.path.join(OUTPUT_DIR, name + '.txt'), np.c_[ew_g, ns_g, ud_g])
    return True, pga


def main():
    parser = argparse.ArgumentParser(description='Convierte K-NET CSV → .txt para CrowdQuake')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='PGA mínimo en g para incluir (default: 0.05)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Convertir todos los archivos sin filtro de PGA')
    args = parser.parse_args()

    threshold = None if args.no_filter else args.threshold
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buscar todas las carpetas EQ-* en DATA/
    eq_dirs = sorted(glob.glob(os.path.join(DATA_DIR, 'EQ-*')))
    if not eq_dirs:
        print(f'No se encontraron carpetas EQ-* en {DATA_DIR}')
        return

    print(f'Carpetas encontradas: {[os.path.basename(d) for d in eq_dirs]}')
    if threshold:
        print(f'Filtro PGA E-W > {threshold}g')
    else:
        print('Sin filtro de PGA')
    print()

    saved = 0
    skipped = 0
    for eq_dir in eq_dirs:
        files = sorted(glob.glob(os.path.join(eq_dir, '*.csv')))
        dir_saved = 0
        for fn in files:
            ok, pga = convert_file(fn, threshold)
            if ok:
                saved += 1
                dir_saved += 1
            else:
                skipped += 1
        print(f'  {os.path.basename(eq_dir)}: {dir_saved}/{len(files)} archivos convertidos')

    print(f'\nTotal: {saved} guardados en EQ/, {skipped} descartados por PGA')
    print(f'Archivos .txt listos en: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
