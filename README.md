# CrowdQuake — Detección Sísmica con Deep Learning

Sistema de detección de terremotos basado en el paper [CrowdQuake (ACM KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403378), adaptado para entrenar con datos K-NET de Japón y ejecutar inferencia en NVIDIA Jetson Nano con acelerómetro ADXL335.

## Estructura del proyecto

```
CROWDQUAKE/
├── convert_knet_csv.py     # Convierte K-NET CSV → .txt para entrenamiento
├── CRNN.py                 # Entrena modelo CRNN (deep learning)
├── ANN.py                  # Entrena modelo ANN baseline (machine learning)
├── run_training.sh         # Ejecuta el pipeline completo
├── test_adxl335.py         # Visualizador en tiempo real del sensor ADXL335
├── DATA/
│   ├── EQ-2024/            # Datos de terremotos K-NET (CSV)
│   ├── EQ-2025-2026/       # Datos de terremotos K-NET (CSV)
│   └── NonEQ/              # Datos de no-terremoto (8 categorías)
├── EQ/                     # Generada automáticamente (.txt convertidos)
├── NonEQ -> DATA/NonEQ     # Symlink
├── result/                 # Modelos entrenados y predicciones
└── Seismic-Detection-master/  # Repositorio original (referencia)
```

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy, joblib

```bash
pip install tensorflow scikit-learn pandas numpy joblib
```

## Datos

### Terremotos (EQ)

Los datos de terremotos se descargan de [NIED K-NET](http://www.kyoshin.bosai.go.jp/kyoshin/data/index_en.html) (requiere registro gratuito).

Pasos para descargar:

1. Ir a NIED → `Download` → `Data Download after Search for Data`
2. Seleccionar `Network` → `K-NET`
3. Seleccionar `Peak acceleration` → `from 50 to 10000` (PGA > 0.05g)
4. Seleccionar rango de fechas (ej: un año completo)
5. Seleccionar formato **CSV**
6. Click `Submit`, seleccionar todos los registros, click `Download All Channels Data`
7. Descomprimir y colocar los CSV en una carpeta dentro de `DATA/`

> NIED muestra máximo 1200 registros por búsqueda. Para descargar más, dividir por rangos de fecha.

Cada carpeta de datos debe seguir la convención `EQ-<periodo>`:

```
DATA/
├── EQ-2022/          # ← nueva carpeta
├── EQ-2023/          # ← nueva carpeta
├── EQ-2024/
├── EQ-2025-2026/
└── NonEQ/
```

El script `convert_knet_csv.py` detecta automáticamente todas las carpetas que empiecen con `EQ-`.

### No-terremotos (NonEQ)

Datos de actividad humana del dataset original de CrowdQuake (sensores IoT a 100 Hz):

| Categoría | Descripción |
|-----------|-------------|
| Bus1, Bus2 | Vibración dentro de autobús |
| DroppingDesk | Golpes en escritorio |
| Floor1 | Vibración de piso |
| Outdoor | Ambiente exterior |
| ShakingDesk | Escritorio vibrando |
| Stay | Sensor estático |
| Walking | Caminando |

Formato: CSV con columnas `dev_id,x,y,z,ts` (aceleración en m/s², timestamp en ms).

## Cómo entrenar

### Opción 1: Pipeline completo (recomendado)

```bash
cd ~/Desktop/PROYECTOS/CROWDQUAKE
bash run_training.sh
```

Esto ejecuta los 3 pasos automáticamente: conversión → CRNN → ANN.

### Opción 2: Paso a paso

```bash
cd ~/Desktop/PROYECTOS/CROWDQUAKE

# 1. Convertir K-NET CSV a formato de entrenamiento
python3 convert_knet_csv.py --threshold 0.05

# 2. Entrenar CRNN
python3 CRNN.py

# 3. Entrenar ANN (opcional, baseline para comparar)
python3 ANN.py
```

### Agregar nuevos datos y re-entrenar

Cuando descargues más datos de NIED:

```bash
# 1. Crear carpeta y colocar los CSV
mkdir -p DATA/EQ-2023
# (copiar los CSV descargados a DATA/EQ-2023/)

# 2. Re-ejecutar el pipeline
bash run_training.sh
```

El script `convert_knet_csv.py` encuentra automáticamente todas las carpetas `EQ-*` en `DATA/` y genera los `.txt` en `EQ/`. Los modelos se re-entrenan con todos los datos disponibles.

### Opciones de conversión

```bash
# PGA > 0.05g (default, más datos)
python3 convert_knet_csv.py --threshold 0.05

# PGA > 0.1g (menos datos, sismos más fuertes)
python3 convert_knet_csv.py --threshold 0.1

# Sin filtro, convertir todo
python3 convert_knet_csv.py --no-filter
```

## Parámetros de entrenamiento

En `CRNN.py` y `ANN.py` se pueden modificar estas variables al inicio del archivo:

```python
SamplingRate = 100    # 25, 50, o 100 Hz (100 = mejor rendimiento)
Duration = 10         # 2, 4, o 10 segundos (10 = mejor rendimiento)
```

La combinación `SamplingRate=100, Duration=10` es la que da mejores resultados según el paper.

## Resultados

### CRNN (158 eventos EQ, 79 archivos NonEQ)

| Métrica | Valor |
|---------|-------|
| AUROC | 0.9990 |
| Accuracy | 99.77% |
| Precision (EQ) | 96.49% |
| Recall (EQ) | 95.37% |
| F1 (EQ) | 95.93% |
| False alarm rate | 0.10% |

### ANN baseline

| Métrica | Valor |
|---------|-------|
| AUROC | 0.9613 |
| Accuracy | 91.43% |
| Precision (EQ) | 23.50% |
| Recall (EQ) | 90.74% |

La CRNN supera ampliamente al ANN. El ANN tiene alta tasa de falsas alarmas.

## Archivos generados

Después de entrenar, en `result/` se encuentran:

| Archivo | Descripción |
|---------|-------------|
| `CRNN_100Hz_10s.h5` | Modelo CRNN entrenado (4 MB) |
| `CRNN_100Hz_10s.csv` | Predicciones del test set |
| `ANN_100Hz_10s.pkl` | Modelo ANN entrenado |
| `ANN_scaler_100Hz_10s.pkl` | Scaler para features del ANN |
| `ANN_100Hz_10s.csv` | Predicciones del test set |

## Formato de datos K-NET CSV

Los archivos descargados de NIED tienen este formato:

```
#K-NET CSV
#Event
#OriginTime,Latitude,Longitude,Depth(km),Magnitude
#2026/01/13 01:58:00,45.048,142.167,0,5.2
#Station
#Code,Latitude,Longitude,Height(m)
#HKD004,45.2149,142.2260,4
#Record
#SamplingFrequency(Hz)
#100
#DurationTime(s)
#162
#Offset
#N-S(gal),E-W(gal),U-D(gal)
#-2.90,10.80,-0.20
#Time,RelativeTime(s),N-S(gal),E-W(gal),U-D(gal)
2026/01/13 01:58:43.00,0.00,-2.97,10.86,-0.20
2026/01/13 01:58:43.01,0.01,-2.97,10.86,-0.20
...
```

`convert_knet_csv.py` extrae las 3 componentes (E-W, N-S, U-D), convierte de gal a g, resta el offset (media), y guarda como `.txt` con 3 columnas separadas por espacio.

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
```

- Ventana de entrada: 2 segundos con overlap de 1 segundo
- Las capas Conv1D actúan como filtros aprendidos (no se necesita filtro bandpass previo)
- El SimpleRNN captura la relación temporal entre las 2 sub-ventanas
- Class weights para compensar el desbalance EQ vs NonEQ

## Referencia

Huang, Lee, Kwon, Lee. *CrowdQuake: A Networked System of Low-Cost Sensors for Earthquake Detection via Deep Learning.* ACM KDD 2020.
