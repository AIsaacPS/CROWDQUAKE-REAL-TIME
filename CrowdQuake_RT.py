#!/usr/bin/env python3
"""
CrowdQuake_RT — Detección Sísmica en Tiempo Real con CRNN
Discrimina sismo vs ruido usando el modelo CrowdQuake CRNN.
Soporta dos fuentes de datos: AnyShake Observer (TCP) y ADXL335 (Serial).

Arquitectura:
  DataSource → RAW Buffer (2s) → STA/LTA sobre RVM filtrado → CRNN (ventana deslizante)

Uso:
  python3 CrowdQuake_RT.py --source anyshake
  python3 CrowdQuake_RT.py --source adxl335 --serial-port /dev/ttyACM0
  python3 CrowdQuake_RT.py --source anyshake --host 192.168.1.100

Referencia: CrowdQuake (ACM KDD 2020) — Huang, Lee, Kwon, Lee
© 2025 SkyAlert de México S.A. de C.V.
"""

import argparse
import logging
import os
import sys
import time
import threading
import socket
import json
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from scipy.signal import butter, lfilter, lfilter_zi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('CrowdQuake_RT')

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes
# ═══════════════════════════════════════════════════════════════════════════════
FS = 100                    # Frecuencia de muestreo objetivo (Hz)
WINDOW_SAMPLES = 200        # Ventana CRNN: 2s @ 100 Hz
ANYSHAKE_COUNT_TO_G = 0.059 / 980.665  # counts → gal → g
G_TO_GAL = 980.665  # Para STA/LTA (necesita gal para rango dinámico adecuado)

# STA/LTA
STA_SEC = 0.3
LTA_SEC = 30
THR_ON = 3.0
THR_OFF = 1.0

# Filtro para STA/LTA (Butterworth bandpass)
FILT_LOW = 0.7
FILT_HIGH = 17.0

# CRNN inferencia
INFERENCE_INTERVAL = 1.0    # Segundos entre inferencias (1/s, ~50% solapamiento)
CONFIRM_CONSEC = 3          # Ventanas CONSECUTIVAS >= threshold para confirmar
TIMEOUT_EVALS = 25          # Timeout sin confirmación (5s)
COOLDOWN_SEC = 10           # Cooldown post-evento

# ADXL335
ADXL_FS = 200               # Frecuencia nativa del ADXL335
ADXL_BAUD = 115200


# ═══════════════════════════════════════════════════════════════════════════════
# Fuentes de datos
# ═══════════════════════════════════════════════════════════════════════════════
class DataSource(ABC):
    """Interfaz común: entrega tuplas (ew, ns, ud) en g @ 100 Hz."""
    @abstractmethod
    def start(self) -> bool:
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def read_samples(self) -> list:
        """Retorna lista de tuplas (ew, ns, ud) en g. Puede estar vacía."""
        ...


class AnyShakeSource(DataSource):
    """Lee datos de AnyShake Observer vía TCP. Sincroniza 3 componentes."""

    def __init__(self, host='localhost', port=30000):
        self.host = host
        self.port = port
        self.sock = None
        self.text_buf = ""
        # Buffers por componente para sincronización
        self._comp = {'ENZ': deque(), 'ENE': deque(), 'ENN': deque()}

    def start(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            self.sock.sendall(b"AT+REALTIME=1\r\n")
            time.sleep(1)
            self.sock.setblocking(False)
            log.info(f"AnyShake conectado: {self.host}:{self.port}")
            return True
        except Exception as e:
            log.error(f"AnyShake conexión fallida: {e}")
            return False

    def stop(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def read_samples(self) -> list:
        # Leer datos TCP disponibles
        raw = b""
        while True:
            try:
                chunk = self.sock.recv(2048)
                if chunk:
                    raw += chunk
                else:
                    break
            except BlockingIOError:
                break
            except Exception:
                break

        if raw:
            self.text_buf += raw.decode('utf-8', errors='ignore')

        # Prevenir crecimiento excesivo
        if len(self.text_buf) > 16384:
            self.text_buf = self.text_buf[-8192:]

        if not self.text_buf:
            return []

        # Parsear líneas completas
        lines = self.text_buf.split('\r')
        self.text_buf = lines[-1]

        for line in lines[:-1]:
            line = line.strip()
            if not line.startswith('$') or '*' not in line:
                continue
            main_part = line.split('*', 1)[0]
            parts = main_part.split(',')
            if len(parts) < 8:
                continue
            comp = parts[4]
            if comp not in self._comp:
                continue
            vals = []
            for p in parts[7:]:
                p = p.strip()
                if p and not p.startswith('*'):
                    try:
                        vals.append(int(p))
                    except ValueError:
                        continue
            if vals:
                self._comp[comp].extend(vals)

        # Sincronizar: emitir tuplas cuando las 3 componentes tienen datos
        out = []
        while all(len(self._comp[c]) > 0 for c in ('ENZ', 'ENE', 'ENN')):
            enz = self._comp['ENZ'].popleft() * ANYSHAKE_COUNT_TO_G
            ene = self._comp['ENE'].popleft() * ANYSHAKE_COUNT_TO_G
            enn = self._comp['ENN'].popleft() * ANYSHAKE_COUNT_TO_G
            # Orden: (EW, NS, UD) = (ENE, ENN, ENZ) — consistente con CRNN training
            out.append((ene, enn, enz))
        return out


class ADXL335Source(DataSource):
    """Lee datos del ADXL335 vía serial. Downsamplea 200→100 Hz."""

    def __init__(self, port='/dev/ttyACM0'):
        self.port = port
        self.ser = None
        self._queue = deque()
        self._thread = None
        self._active = False
        self._skip = False  # Para downsample 2:1

    def start(self) -> bool:
        try:
            import serial
            self.ser = serial.Serial(self.port, ADXL_BAUD, timeout=0.1)
            self.ser.reset_input_buffer()
            log.info(f"ADXL335 conectado: {self.port}")
        except Exception as e:
            log.error(f"ADXL335 conexión fallida: {e}")
            return False
        self._active = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _read_loop(self):
        buf = bytearray()
        while self._active:
            try:
                data = self.ser.read(512)
                if not data:
                    time.sleep(0.001)
                    continue
                for byte in data:
                    if byte == ord(b'\n'):
                        line = buf.decode(errors='ignore').strip()
                        buf.clear()
                        if line:
                            self._parse_line(line)
                    else:
                        buf.append(byte)
                        if len(buf) > 512:
                            buf.clear()
            except Exception:
                time.sleep(0.01)

    def _parse_line(self, line):
        parts = line.split(',')
        if len(parts) != 4:
            return
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return
        # Downsample 200→100 Hz: tomar 1 de cada 2
        self._skip = not self._skip
        if self._skip:
            return
        # ADXL335: x=EW, y=NS, z=UD (ya en g)
        self._queue.append((x, y, z))

    def read_samples(self) -> list:
        out = list(self._queue)
        self._queue.clear()
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# STA/LTA recursivo
# ═══════════════════════════════════════════════════════════════════════════════
class RecursiveSTA_LTA:
    def __init__(self, nsta=int(STA_SEC * FS), nlta=int(LTA_SEC * FS)):
        self.csta = 1.0 / nsta
        self.clta = 1.0 / nlta
        self.sta = 0.0
        self.lta = 1.0
        self.n = 0
        self.warmup = nlta  # Necesita llenar LTA antes de ser confiable

    def update(self, value) -> float:
        self.n += 1
        self.sta = self.csta * value + (1.0 - self.csta) * self.sta
        self.lta = self.clta * value + (1.0 - self.clta) * self.lta
        if self.n < self.warmup or self.lta < 1e-10:
            return 0.0
        return self.sta / self.lta


# ═══════════════════════════════════════════════════════════════════════════════
# Filtro Butterworth para STA/LTA
# ═══════════════════════════════════════════════════════════════════════════════
class BandpassFilter:
    """Butterworth bandpass aplicado muestra a muestra."""
    def __init__(self, low=FILT_LOW, high=FILT_HIGH, fs=FS, order=3):
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype='bandpass')
        self.b, self.a = b, a
        self.zi = lfilter_zi(b, a).astype(np.float64)

    def process(self, sample: float) -> float:
        out, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi)
        return out[0]


# ═══════════════════════════════════════════════════════════════════════════════
# CRNN Wrapper
# ═══════════════════════════════════════════════════════════════════════════════
class CrowdQuakeCRNN:
    def __init__(self, model_path):
        log.info(f"Cargando modelo CRNN: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # Warmup
        dummy = np.zeros((1, 2, 100, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        log.info(f"CRNN listo (input={self.model.input_shape}, params={self.model.count_params():,})")

    def predict(self, buffer_200x3: np.ndarray) -> float:
        """buffer_200x3: array (200, 3) en g. Retorna probabilidad [0,1]."""
        # Reshape: (200,3) → (1, 2, 100, 3)
        x = buffer_200x3.reshape(1, 2, 100, 3).astype(np.float32)
        prob = self.model.predict(x, verbose=0)[0, 0]
        return float(prob)


# ═══════════════════════════════════════════════════════════════════════════════
# Detector principal
# ═══════════════════════════════════════════════════════════════════════════════
class CrowdQuakeRT:
    def __init__(self, source: DataSource, model_path: str, threshold=0.5):
        self.source = source
        self.crnn = CrowdQuakeCRNN(model_path)
        self.threshold = threshold

        # Buffer RAW: últimas 200 muestras (2s) sin filtrar, 3 ejes
        self.raw_buf = deque(maxlen=WINDOW_SAMPLES)

        # Filtros bandpass para STA/LTA (uno por eje)
        self.filt_ew = BandpassFilter()
        self.filt_ns = BandpassFilter()
        self.filt_ud = BandpassFilter()

        # STA/LTA sobre RVM filtrado
        self.stalta = RecursiveSTA_LTA()

        # Estado
        self.anomaly_active = False
        self.anomaly_start = None
        self.last_inference = 0.0
        self.consec_count = 0
        self.eval_count = 0
        self.confirmed = False
        self.cooldown_until = 0.0
        self.event_count = 0
        self.sample_count = 0

        # Estadísticas
        self.total_inferences = 0

        os.makedirs("events", exist_ok=True)

    def run(self):
        """Loop principal."""
        if not self.source.start():
            log.error("No se pudo iniciar la fuente de datos")
            return

        log.info("="*60)
        log.info("CrowdQuake_RT — Detección en tiempo real")
        log.info(f"  Fuente: {self.source.__class__.__name__}")
        log.info(f"  Modelo: CRNN ({self.crnn.model.count_params():,} params)")
        log.info(f"  Umbral CRNN: {self.threshold}")
        log.info(f"  STA/LTA: {STA_SEC}s/{LTA_SEC}s, ON={THR_ON}, OFF={THR_OFF}")
        log.info(f"  Inferencia cada {INFERENCE_INTERVAL}s durante anomalía")
        log.info(f"  Confirmación: {CONFIRM_CONSEC} ventanas consecutivas >= umbral")
        log.info("="*60)
        log.info("Esperando datos... (warmup STA/LTA: ~30s)")

        try:
            while True:
                samples = self.source.read_samples()
                if not samples:
                    time.sleep(0.005)
                    continue

                for ew, ns, ud in samples:
                    self._process_sample(ew, ns, ud)

        except KeyboardInterrupt:
            log.info("Detenido por usuario")
        finally:
            self.source.stop()
            log.info(f"Total: {self.sample_count} muestras, "
                     f"{self.total_inferences} inferencias, "
                     f"{self.event_count} sismos confirmados")

    def _process_sample(self, ew, ns, ud):
        self.sample_count += 1

        # 1. Almacenar en buffer RAW (sin filtrar)
        self.raw_buf.append((ew, ns, ud))

        # 2. Filtrar para STA/LTA (en gal para rango dinámico adecuado)
        f_ew = self.filt_ew.process(ew * G_TO_GAL)
        f_ns = self.filt_ns.process(ns * G_TO_GAL)
        f_ud = self.filt_ud.process(ud * G_TO_GAL)

        # 3. RVM filtrado
        rvm = np.sqrt(f_ew**2 + f_ns**2 + f_ud**2)

        # 4. STA/LTA
        ratio = self.stalta.update(rvm * rvm)  # Energía (cuadrado)

        now = time.time()

        # 5. Lógica de estados
        if not self.anomaly_active:
            # ¿Iniciar anomalía?
            if ratio >= THR_ON and now > self.cooldown_until:
                self.anomaly_active = True
                self.anomaly_start = now
                self.consec_count = 0
                self.eval_count = 0
                self.confirmed = False
                self.last_inference = 0.0
                utc = datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]
                log.warning(f"ANOMALÍA DETECTADA | UTC={utc} | ratio={ratio:.1f}")
        else:
            # ¿Fin de anomalía?
            if ratio <= THR_OFF:
                dur = now - self.anomaly_start
                utc = datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]
                log.info(f"Anomalía finalizada | UTC={utc} | dur={dur:.1f}s | "
                         f"evals={self.eval_count} | confirmado={self.confirmed}")
                self.anomaly_active = False
                if self.confirmed:
                    self.cooldown_until = now + COOLDOWN_SEC
                return

            # Inferencia CRNN cada INFERENCE_INTERVAL
            if (now - self.last_inference) >= INFERENCE_INTERVAL and len(self.raw_buf) >= WINDOW_SAMPLES:
                self._run_inference(now)

            # Timeout sin confirmación
            if not self.confirmed and self.eval_count >= TIMEOUT_EVALS:
                log.info(f"Timeout CRNN ({self.eval_count} evals sin confirmación)")
                # No desactivar anomalía — STA/LTA la controla
                # Solo dejar de evaluar
                self.eval_count = 0
                self.consec_count = 0

    def _run_inference(self, now):
        self.last_inference = now
        self.eval_count += 1
        self.total_inferences += 1

        # Snapshot del buffer RAW → array (200, 3)
        buf = np.array(list(self.raw_buf), dtype=np.float32)

        # Restar media por eje (como en entrenamiento)
        buf -= buf.mean(axis=0)

        # Inferencia
        t0 = time.perf_counter()
        prob = self.crnn.predict(buf)
        dt_ms = (time.perf_counter() - t0) * 1000

        # Contador de consecutivas
        if prob >= self.threshold:
            self.consec_count += 1
        else:
            self.consec_count = 0

        # PGA del buffer actual (max abs en g)
        pga_g = np.max(np.abs(buf))
        pga_gal = pga_g * 980.665

        status = "🔴" if prob >= self.threshold else "⚪"
        log.info(f"  CRNN {status} prob={prob:.4f} | consec={self.consec_count}/{CONFIRM_CONSEC} | "
                 f"PGA={pga_gal:.1f}gal | {dt_ms:.0f}ms")

        # Confirmación: N ventanas consecutivas
        if not self.confirmed and self.consec_count >= CONFIRM_CONSEC:
            self.confirmed = True
            self.event_count += 1
            utc = datetime.now(timezone.utc)
            utc_str = utc.strftime('%H:%M:%S.%f')[:-3]
            latency = now - self.anomaly_start

            log.critical(f"🚨 SISMO CONFIRMADO #{self.event_count} | UTC={utc_str} | "
                         f"prob={prob:.3f} | PGA={pga_gal:.1f}gal | latencia={latency:.2f}s")

            self._save_event(utc, prob, pga_gal, latency)

    def _save_event(self, utc, prob, pga_gal, latency):
        """Guarda evento confirmado en JSON."""
        event = {
            "event_id": self.event_count,
            "timestamp_utc": utc.isoformat() + "Z",
            "system": "CrowdQuake_RT",
            "model": "CRNN_100Hz_10s",
            "source": self.source.__class__.__name__,
            "crnn_probability": round(prob, 4),
            "crnn_threshold": self.threshold,
            "pga_gals": round(pga_gal, 2),
            "detection_latency_s": round(latency, 3),
            "stalta_params": {"sta": STA_SEC, "lta": LTA_SEC, "thr_on": THR_ON, "thr_off": THR_OFF},
            "inference_interval_s": INFERENCE_INTERVAL,
            "confirmation": f"{CONFIRM_CONSEC} consecutive >= {self.threshold}"
        }
        fname = f"events/crowdquake_{utc.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(fname, 'w') as f:
                json.dump(event, f, indent=2)
            log.info(f"Evento guardado: {fname}")
        except Exception as e:
            log.error(f"Error guardando evento: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='CrowdQuake_RT — Detección sísmica en tiempo real con CRNN')
    parser.add_argument('--source', choices=['anyshake', 'adxl335'], default='anyshake',
                        help='Fuente de datos (default: anyshake)')
    parser.add_argument('--host', default='localhost',
                        help='Host de AnyShake Observer (default: localhost)')
    parser.add_argument('--port', type=int, default=30000,
                        help='Puerto de AnyShake Observer (default: 30000)')
    parser.add_argument('--serial-port', default='/dev/ttyACM0',
                        help='Puerto serial del ADXL335 (default: /dev/ttyACM0)')
    parser.add_argument('--model', default='result/CRNN_100Hz_10s.h5',
                        help='Ruta al modelo CRNN (default: result/CRNN_100Hz_10s.h5)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Umbral de probabilidad CRNN (default: 0.9)')
    args = parser.parse_args()

    # Resolver ruta del modelo relativa al script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)

    if not os.path.exists(model_path):
        log.error(f"Modelo no encontrado: {model_path}")
        sys.exit(1)

    # Crear fuente de datos
    if args.source == 'anyshake':
        source = AnyShakeSource(host=args.host, port=args.port)
    else:
        source = ADXL335Source(port=args.serial_port)

    detector = CrowdQuakeRT(source, model_path, threshold=args.threshold)
    detector.run()


if __name__ == '__main__':
    main()
