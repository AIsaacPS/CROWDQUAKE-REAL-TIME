#!/usr/bin/env python3
"""
Test ADXL335 + SKYACC - Visualizador en tiempo real
Usa el mismo pipeline de lectura serial que SKYALERT-AI-EQDetector.py
para validar que el sketch de Arduino produce datos correctos.

Uso:
    python test_adxl335.py                    # Puerto por defecto /dev/ttyACM0
    python test_adxl335.py /dev/ttyUSB0       # Puerto específico
    python test_adxl335.py COM3               # Windows
"""

import sys
import time
import threading
import numpy as np
import serial
from collections import deque
from queue import Queue
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===== Parámetros idénticos a SKYALERT-AI-EQDetector.py =====
FS = 200.0
LOW_CUT = 0.1
HIGH_CUT = 10.0
G_TO_GAL = 98.0665  # Mismo factor que SKYACC_G_TO_GAL en el detector

VENTANA_SEG = 15
BUFFER_TAM = int(FS * VENTANA_SEG)


class SKYACCLectorSerial:
    """Copia exacta del lector serial del detector principal"""
    def __init__(self, puerto, queue, nombre="ADXL335"):
        self.puerto = puerto
        self.queue = queue
        self.nombre = nombre
        self.ser = None
        self.activo = False
        self.buffer = bytearray()
        self.ok = 0
        self.corruptas = 0
        self.lock = threading.Lock()
        self.hilo = None

        sos = butter(N=4, Wn=[LOW_CUT, HIGH_CUT], btype='band', fs=FS, output='sos')
        self.sos = sos
        self.zi = [sosfilt_zi(sos) for _ in range(3)]

    def iniciar(self):
        try:
            self.ser = serial.Serial(self.puerto, 115200, timeout=0.1)
            self.ser.reset_input_buffer()
            print(f"[{self.nombre}] Conectado: {self.puerto}")
        except Exception as e:
            print(f"[{self.nombre}] Error: {e}")
            return False
        self.activo = True
        self.hilo = threading.Thread(target=self._leer_loop, daemon=True)
        self.hilo.start()
        return True

    def detener(self):
        self.activo = False
        if self.hilo:
            self.hilo.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _leer_loop(self):
        while self.activo:
            try:
                datos = self.ser.read(512)
                if datos:
                    for byte in datos:
                        if byte == ord(b'\n'):
                            linea = self.buffer.decode(errors='ignore').strip()
                            self.buffer.clear()
                            if linea:
                                self._procesar(linea)
                        else:
                            self.buffer.append(byte)
                            if len(self.buffer) > 512:
                                self.buffer.clear()
                                with self.lock:
                                    self.corruptas += 1
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[{self.nombre}] Error lectura: {e}")
                time.sleep(0.1)

    def _procesar(self, linea):
        partes = linea.split(',')
        if len(partes) == 4:
            try:
                t = float(partes[0])
                vals = [float(partes[i]) for i in range(1, 4)]

                filt = []
                for i, v in enumerate(vals):
                    f, self.zi[i] = sosfilt(self.sos, [v], zi=self.zi[i])
                    filt.append(f[0] * G_TO_GAL)

                with self.lock:
                    self.ok += 1
                self.queue.put((t, *filt))
                return
            except Exception:
                pass
        with self.lock:
            self.corruptas += 1
        if self.corruptas <= 5:
            print(f"[{self.nombre}] Línea corrupta: {repr(linea)}")

    def stats(self):
        with self.lock:
            return self.ok, self.corruptas


def main():
    puerto = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'

    queue = Queue()
    lector = SKYACCLectorSerial(puerto, queue)
    if not lector.iniciar():
        sys.exit(1)

    # Buffers de visualización
    t_buf = deque(maxlen=BUFFER_TAM)
    x_buf = deque(maxlen=BUFFER_TAM)
    y_buf = deque(maxlen=BUFFER_TAM)
    z_buf = deque(maxlen=BUFFER_TAM)
    t0 = None
    lock = threading.Lock()

    # Hilo consumidor de la queue
    def consumir():
        nonlocal t0
        while lector.activo:
            try:
                t_ms, x, y, z = queue.get(timeout=0.1)
                with lock:
                    if t0 is None:
                        t0 = t_ms
                    t_buf.append((t_ms - t0) / 1000.0)
                    x_buf.append(x)
                    y_buf.append(y)
                    z_buf.append(z)
            except Exception:
                pass

    hilo_consumir = threading.Thread(target=consumir, daemon=True)
    hilo_consumir.start()

    # Visualización
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'ADXL335 Test | Filtro {LOW_CUT}-{HIGH_CUT} Hz | {puerto}')

    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    nombres = ['X', 'Y', 'Z']
    lineas = []
    textos = []

    for i, ax in enumerate(axes):
        line, = ax.plot([], [], color=colores[i], lw=1)
        lineas.append(line)
        ax.set_ylabel(f'{nombres[i]} (Gal)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        txt = ax.text(0.01, 0.95, '', transform=ax.transAxes, fontsize=9,
                      va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        textos.append(txt)

    axes[-1].set_xlabel('Tiempo (s)')
    info_txt = fig.text(0.01, 0.01, '', fontsize=9,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.9))

    last_stats = [time.time()]

    def actualizar(_):
        with lock:
            if len(t_buf) < 2:
                return lineas + textos + [info_txt]
            t = np.array(t_buf)
            datos = [np.array(x_buf), np.array(y_buf), np.array(z_buf)]

        for i, (line, d) in enumerate(zip(lineas, datos)):
            line.set_data(t, d)
            max_abs = max(np.max(np.abs(d)), 0.5)
            axes[i].set_ylim(-max_abs * 1.3, max_abs * 1.3)

        xmin = max(0, t[-1] - VENTANA_SEG)
        xmax = max(VENTANA_SEG, t[-1] + 0.5)
        axes[0].set_xlim(xmin, xmax)

        now = time.time()
        if now - last_stats[0] > 0.5:
            last_stats[0] = now
            ok, corr = lector.stats()
            for i, d in enumerate(datos):
                textos[i].set_text(f'Máx: {np.max(np.abs(d)):.2f} Gal')
            freq = ok / max(t[-1], 1)
            info_txt.set_text(f'Muestras: {ok} | Corruptas: {corr} | '
                              f'Freq: {freq:.0f} Hz | Tiempo: {t[-1]:.1f}s')

        return lineas + textos + [info_txt]

    ani = animation.FuncAnimation(fig, actualizar, interval=50, blit=False)

    def on_close(_):
        lector.detener()
        ok, corr = lector.stats()
        print(f"\nFinalizado. OK={ok}, Corruptas={corr}")

    fig.canvas.mpl_connect('close_event', on_close)

    print(f"\nEsperando datos del ADXL335 en {puerto}...")
    print("Cierra la ventana o Ctrl+C para salir.\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        lector.detener()


if __name__ == '__main__':
    main()
