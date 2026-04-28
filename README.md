# CrowdQuake — Detección Sísmica en Tiempo Real con Deep Learning

Sistema de detección de terremotos basado en el paper [CrowdQuake (ACM KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403378), adaptado para entrenar con datos K-NET de Japón y ejecutar detección en tiempo real en NVIDIA Jetson Orin Nano con AnyShake Observer o acelerómetro ADXL335.

**Objetivo:** Discriminar sismos de ruido con alta precisión para alimentar un sistema de alerta sísmica temprana de SkyAlert.

## Estructura del proyecto

```
CROWDQUAKE-REAL-TIME-main/
│
├── CRNN.py                 # Entrenamiento del modelo CRNN (deep learning)
├── ANN.py                  # Entrenamiento del modelo ANN baseline
├── convert_knet_csv.py     # Convierte K-NET CSV → .txt para entrenamiento
├── run_training.sh         # Pipeline completo automatizado
├── plot_results.py         # Genera curvas ROC, PR, matrices de confusión
│
├── CrowdQuake_RT.py        # ★ Sistema de detección en TIEMPO REAL
├── simulador_CRNN.py       # ★ Validador: ventana deslizante sobre datos reales
├── capture_noneq.py        # ★ Captura datos de ruido del AnyShake para entrenamiento
├── test_adxl335.py         # Visualizador en tiempo real del sensor ADXL335
│
├── CSV-Data/
│   ├── EQ-2011-2012/       # 458 registros K-NET
│   ├── EQ-2013-2014/       # 85 registros
│   ├── EQ-2015-2016/       # 259 registros
│   ├── EQ-2017-2018/       # 130 registros
│   ├── EQ-2019-2020/       # 96 registros
│   ├── EQ-2021-2022/       # 117 registros
│   ├── EQ-2023/            # 26 registros
│   ├── EQ-2024/            # 161 registros
│   ├── EQ-2025-2026/       # 101 registros
│   └── NonEQ/              # 82 archivos (11 categorías de ruido)
│       ├── knuEqAccData_Bus1/          # 9 archivos — Vibración autobús
│       ├── knuEqAccData_Bus2/          # 10 archivos — Vibración autobús
│       ├── knuEqAccData_DroppingDesk/  # 10 archivos — Golpes en escritorio
│       ├── knuEqAccData_Floor1/        # 10 archivos — Vibración de piso
│       ├── knuEqAccData_Outdoor/       # 10 archivos — Ambiente exterior
│       ├── knuEqAccData_ShakingDesk/   # 10 archivos — Escritorio vibrando
│       ├── knuEqAccData_Stay/          # 10 archivos — Sensor estático
│       ├── knuEqAccData_Walking/       # 10 archivos — Caminando
│       ├── knuEqAccData_AnyShake_Shaking/  # ★ Sacudidas del sensor
│       ├── knuEqAccData_AnyShake_Tapping/  # ★ Golpes al sensor
│       └── knuEqAccData_AnyShake_Surface/  # ★ Vibraciones de superficie
│
├── EQ/                     # .txt convertidos (generado por convert_knet_csv.py)
├── NonEQ -> CSV-Data/NonEQ # Symlink
│
├── result/
│   ├── CRNN_100Hz_10s.h5   # ★ Modelo CRNN entrenado (4 MB, 340K params)
│   ├── CRNN_100Hz_10s.csv  # Predicciones del test set
│   ├── ANN_100Hz_10s.pkl   # Modelo ANN baseline
│   ├── ANN_scaler_100Hz_10s.pkl
│   ├── ANN_100Hz_10s.csv
│   ├── evaluacion_modelos.png
│   ├── distribucion_probabilidades.png
│   └── REALTIME-RESULTS/   # ★ Gráficos de validación en tiempo real
│       ├── EQ-2023/
│       ├── EQ-2024/
│       ├── EQ-2025-2026/
│       ├── knuEqAccData_Bus1/
│       ├── knuEqAccData_Bus2/
│       ├── knuEqAccData_DroppingDesk/
│       ├── knuEqAccData_Floor1/
│       ├── knuEqAccData_Outdoor/
│       ├── knuEqAccData_ShakingDesk/
│       ├── knuEqAccData_Stay/
│       ├── knuEqAccData_Walking/
│       ├── knuEqAccData_AnyShake_Shaking/
│       ├── knuEqAccData_AnyShake_Tapping/
│       └── knuEqAccData_AnyShake_Surface/
│
├── events/                 # Eventos detectados en tiempo real (JSON)
├── docs/                   # Documentación y referencias
│   ├── CrowdQuake - Networked System of...pdf  # Paper original (KDD 2020)
│   ├── PROPUESTA DE IMPLEMENTACION DE CROWDQUAKE EN JETSON NANO.txt
│   ├── README-oficial-CROWDQUAKE.md            # README original del repo
│   └── comparacion_reentrenamiento.txt         # Histórico de métricas v1→v2
└── Seismic-Detection-master/  # Código original CrowdQuake (referencia)
```

## Estado del proyecto

### ✅ Fase 1 — Entrenamiento (completada)

Pipeline automatizado de entrenamiento con 1,433 registros K-NET (2011-2026):

- **680 eventos sísmicos** pasan el filtro PGA > 0.05g → 6,120 ventanas de entrenamiento
- **82 archivos de ruido** en 11 categorías (8 CrowdQuake IoT + 3 AnyShake) → 50,629 ventanas
- Modelo CRNN entrenado con 100 epochs, class weights para balanceo

### ✅ Fase 2 — Sistema en tiempo real (completada)

`CrowdQuake_RT.py` — Detector en tiempo real con dos fuentes de datos:

- **AnyShake Observer** (TCP, 100 Hz)
- **ADXL335** (Serial, 200 Hz → downsample a 100 Hz)

Arquitectura de detección en dos niveles:
1. **STA/LTA** sobre RVM filtrado (Butterworth 0.7-17 Hz) → trigger de anomalía
2. **CRNN** con ventana deslizante de 2s → confirmación: **3 ventanas consecutivas ≥ 0.9**

### ✅ Fase 3 — Validación (completada)

`simulador_CRNN.py` — Validador que desliza la ventana CRNN sobre trazas sísmicas reales y genera gráficos de señal vs. probabilidad. Criterio de confirmación idéntico al sistema en tiempo real.

### ✅ Fase 4 — Captura de ruido ambiental y re-entrenamiento (completada)

`capture_noneq.py` — Captura datos de ruido del AnyShake para mejorar la discriminación del modelo. Se capturaron 3 categorías de ruido mecánico (sacudidas, golpes, vibraciones de superficie) y se re-entrenó el modelo, reduciendo falsas alarmas por manipulación del sensor.

## Resultados del modelo CRNN

### Entrenamiento actual (680 EQ + 82 NonEQ, 28 abril 2026)

| Métrica | CRNN | ANN (baseline) |
|---------|------|----------------|
| **AUROC** | **0.9998** | 0.9476 |
| **Accuracy** | **99.79%** | 74.39% |
| **Precision (EQ)** | **99.18%** | 29.37% |
| **Recall (EQ)** | **98.91%** | 95.21% |
| **F1 (EQ)** | **99.05%** | 44.89% |

### Evolución del modelo

| Métrica | v1 (158 EQ, 79 NonEQ) | v2 (680 EQ, 79 NonEQ) | v3 (680 EQ, 82 NonEQ) |
|---------|------------------------|------------------------|------------------------|
| AUROC | 0.9990 | 0.9998 | **0.9998** |
| Precision (EQ) | 96.49% | 98.38% | **99.18%** |
| Recall (EQ) | 95.37% | 99.13% | **98.91%** |
| F1 (EQ) | 95.93% | 98.75% | **99.05%** |

v3 mejoró la Precision (+0.80%) al aprender a rechazar ruido mecánico del AnyShake, con pérdida mínima de Recall (-0.22%).

### Validación con simulador (umbral 0.9, 3 consecutivas)

| Dataset | Resultado | Métrica |
|---------|-----------|---------|
| EQ-2025-2026 (101 archivos) | 101/101 | **Recall = 100%** |
| EQ-2024 (161 archivos) | 161/161 | **Recall = 100%** |
| EQ-2023 (26 archivos) | 25/26 | **Recall = 96.2%** |
| NonEQ completo (82 archivos) | 10/82 FA | **Especificidad = 87.8%** |

Falsas alarmas por categoría NonEQ:

| Categoría | Archivos | FA | Especificidad |
|-----------|----------|----|---------------|
| Bus1 | 9 | 2 | 77.8% |
| Bus2 | 10 | 3 | 70.0% |
| DroppingDesk | 10 | 1 | 90.0% |
| Floor1 | 10 | 0 | **100%** |
| Outdoor | 10 | 0 | **100%** |
| ShakingDesk | 10 | 1 | 90.0% |
| Stay | 10 | 0 | **100%** |
| Walking | 10 | 3 | 70.0% |
| AnyShake_Shaking | 1 | 0 | **100%** |
| AnyShake_Tapping | 1 | 1 | 0% |
| AnyShake_Surface | 1 | 0 | **100%** |

## Uso

### Entrenamiento

```bash
# Pipeline completo (conversión → CRNN → ANN)
bash run_training.sh

# O paso a paso
python3 convert_knet_csv.py --threshold 0.05
python3 CRNN.py
python3 ANN.py
python3 plot_results.py
```

### Detección en tiempo real

```bash
# Con AnyShake Observer (umbral 0.9 por defecto)
python3 CrowdQuake_RT.py --source anyshake

# Con AnyShake en host remoto
python3 CrowdQuake_RT.py --source anyshake --host 192.168.1.100 --port 30000

# Con ADXL335
python3 CrowdQuake_RT.py --source adxl335 --serial-port /dev/ttyACM0

# Ajustar umbral de detección
python3 CrowdQuake_RT.py --source anyshake --threshold 0.95
```

El sistema confirma un sismo cuando **3 ventanas consecutivas** tienen probabilidad CRNN ≥ 0.9. Los eventos confirmados se guardan en `events/` como JSON.

### Validación con datos reales

```bash
# Validar un archivo específico
python3 simulador_CRNN.py CSV-Data/EQ-2024/ISK0062401011618.csv

# Validar carpeta completa de sismos
python3 simulador_CRNN.py CSV-Data/EQ-2025-2026/

# Validar una categoría de ruido
python3 simulador_CRNN.py CSV-Data/NonEQ/knuEqAccData_AnyShake_Shaking/

# Validar todo el ruido (ejecutar por categoría para evitar sobreescritura de PNGs)
for d in CSV-Data/NonEQ/knuEqAccData_*/; do python3 simulador_CRNN.py "$d"; done

# Ajustar umbral
python3 simulador_CRNN.py DATA/EQ-2023/ --threshold 0.95
```

Los gráficos se guardan en `result/REALTIME-RESULTS/<nombre_dataset>/`.

### Captura de datos NonEQ desde AnyShake

`capture_noneq.py` graba datos del AnyShake en formato compatible con el entrenamiento. Esto permite mejorar el modelo capturando ruido real del entorno de despliegue.

```bash
# Capturar 5 minutos de sacudidas del sensor
python3 capture_noneq.py --category Shaking --duration 5

# Capturar 10 minutos de golpes
python3 capture_noneq.py --category Tapping --duration 10

# Capturar 60 minutos de ruido ambiental diurno
python3 capture_noneq.py --category Day --duration 60

# Capturar con AnyShake en host remoto
python3 capture_noneq.py --category Traffic --duration 30 --host 192.168.1.100
```

Los archivos se guardan en `CSV-Data/NonEQ/knuEqAccData_AnyShake_<categoría>/` en CSVs de 5 minutos cada uno, con formato idéntico al dataset CrowdQuake original.

**Flujo para mejorar el modelo con datos nuevos:**

```bash
# 1. Capturar ruido del entorno (sensor fijo, condiciones reales)
python3 capture_noneq.py --category MiEntorno --duration 60

# 2. Re-entrenar el modelo (incluye automáticamente los nuevos datos)
python3 CRNN.py

# 3. Validar que no se perdió recall en sismos
python3 simulador_CRNN.py CSV-Data/EQ-2025-2026/

# 4. Validar que el nuevo ruido ya no genera falsas alarmas
python3 simulador_CRNN.py CSV-Data/NonEQ/knuEqAccData_AnyShake_MiEntorno/

# 5. Si los resultados son satisfactorios, el modelo está listo
#    El sistema en tiempo real usa automáticamente result/CRNN_100Hz_10s.h5
```

**Categorías de ruido recomendadas para capturar en campo:**

| Categoría | Descripción | Duración sugerida |
|-----------|-------------|-------------------|
| `Quiet` | Noche/madrugada, mínima actividad | 60 min |
| `Day` | Día normal, actividad humana habitual | 60-120 min |
| `Traffic` | Hora pico, máximo tráfico vehicular | 30-60 min |
| `Wind` | Día con viento fuerte | 30 min |
| `Shaking` | Sacudidas intencionales del sensor | 5-10 min |
| `Tapping` | Golpes en la superficie cercana | 5-10 min |
| `Construction` | Obras o maquinaria cercana | 30 min |

Mientras más variado sea el ruido capturado, mejor discriminará el modelo entre sismos reales y perturbaciones ambientales.

## Arquitectura del sistema en tiempo real

```
Fuente de datos (AnyShake TCP ó ADXL335 Serial)
        │
        ▼
  DataSource (interfaz común)
   ├── AnyShakeSource  ← TCP, parsea $...*XX, sincroniza 3 componentes
   └── ADXL335Source   ← Serial 115200, parsea t,x,y,z, downsample 200→100Hz
        │
        ▼  Muestras sincronizadas (EW, NS, UD) en g @ 100 Hz
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 RAW Buf   Filtro Butterworth 0.7-17 Hz (por eje)
 (2s)        │
   │         ▼
   │      RVM = √(ew² + ns² + ud²)  [en gal]
   │         │
   │         ▼
   │      STA/LTA recursivo (0.3s / 30s)
   │         │
   │         ▼
   │      ratio > 3.0 → ANOMALÍA ACTIVA
   │         │
   ▼         │
 CRNN ◄──────┘  (solo durante anomalía)
   │
   ▼
 prob ≥ 0.9 × 3 consecutivas → SISMO CONFIRMADO
```

### Ventana deslizante CRNN

- **Ventana**: 2 segundos (200 muestras @ 100 Hz, 3 ejes)
- **Slide**: 1 segundo (~50% solapamiento)
- **Tensor de entrada**: (1, 2, 100, 3) — 2 sub-ventanas de 1s
- **Confirmación**: 3 ventanas consecutivas con probabilidad ≥ 0.9
- **Datos**: RAW sin filtrar, media restada por eje (como en entrenamiento)

### Latencia de detección

| Componente | Tiempo |
|---|---|
| STA/LTA trigger | continuo |
| Primera inferencia CRNN | 0ms (buffer ya tiene 2s) |
| Inferencia en Jetson Orin Nano | ~10ms |
| Confirmación (3 consecutivas @ 1s) | 2-3s |
| **Total post-trigger** | **~2-3s** |

## Arquitectura CRNN

```
Input: (batch, 2, 100, 3)  →  2 sub-ventanas de 1s, 3 ejes (EW, NS, UD)
  → TimeDistributed(Conv1D(64, kernel=3, relu))
  → TimeDistributed(Conv1D(64, kernel=3, relu))
  → TimeDistributed(Dropout(0.5))
  → TimeDistributed(MaxPooling1D(2))
  → TimeDistributed(Flatten())
  → SimpleRNN(100)
  → Dropout(0.5)
  → Dense(100, relu)
  → Dense(1, sigmoid)
Output: probabilidad [0, 1] de terremoto

Total: 340,493 parámetros (4 MB)
```

## Datos

### Terremotos (EQ)

1,433 registros K-NET de [NIED](http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html) (2011-2026), de los cuales 680 pasan el filtro PGA > 0.05g.

Para descargar más datos:
1. Registrarse en NIED K-NET
2. `Download` → `Data Download after Search for Data`
3. `Network` → `K-NET`, `Peak acceleration` → `from 50 to 10000`
4. Seleccionar rango de fechas, formato CSV, descargar
5. Colocar en `CSV-Data/EQ-<periodo>/` y re-ejecutar `bash run_training.sh`

### No-terremotos (NonEQ)

82 archivos en 11 categorías:

| Categoría | Archivos | Fuente | Descripción |
|-----------|----------|--------|-------------|
| Bus1 | 9 | CrowdQuake IoT | Vibración dentro de autobús |
| Bus2 | 10 | CrowdQuake IoT | Vibración dentro de autobús |
| DroppingDesk | 10 | CrowdQuake IoT | Golpes en escritorio |
| Floor1 | 10 | CrowdQuake IoT | Vibración de piso |
| Outdoor | 10 | CrowdQuake IoT | Ambiente exterior |
| ShakingDesk | 10 | CrowdQuake IoT | Escritorio vibrando |
| Stay | 10 | CrowdQuake IoT | Sensor estático |
| Walking | 10 | CrowdQuake IoT | Caminando |
| AnyShake_Shaking | 1 | AnyShake local | Sacudidas del sensor |
| AnyShake_Tapping | 1 | AnyShake local | Golpes al sensor |
| AnyShake_Surface | 1 | AnyShake local | Vibraciones de superficie |

Para agregar más datos NonEQ del AnyShake, ver la sección [Captura de datos NonEQ desde AnyShake](#captura-de-datos-noneq-desde-anyshake).

## Requisitos

- Python 3.8+
- TensorFlow 2.x (2.11.0 verificado)
- scikit-learn, pandas, numpy, scipy, matplotlib
- pyserial (solo para ADXL335)

```bash
pip install tensorflow scikit-learn pandas numpy scipy matplotlib pyserial
```

### Hardware verificado

- NVIDIA Jetson Orin Nano (Ubuntu 20.04, CUDA 11.4, cuDNN 8.6)
- Modelo CRNN: 4 MB, ~10ms inferencia en GPU

## Referencia

Huang, Lee, Kwon, Lee. *CrowdQuake: A Networked System of Low-Cost Sensors for Earthquake Detection via Deep Learning.* ACM KDD 2020.

## Licencia

© 2025 SkyAlert de México S.A. de C.V. — Todos los derechos reservados.
