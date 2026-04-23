#!/usr/bin/env python3
"""
====================================
Implementation of CRNN (TF2 compatible)
====================================
Basado en el código original de CrowdQuake (KDD 2020).
Adaptado para TensorFlow 2.x y estructura de carpetas local.
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

# =========================== Configuración ===================================
SamplingRate = 100       # 25, 50, o 100 Hz
Duration = 10            # 2, 4, o 10 segundos (10 = mejor rendimiento según paper)

work_path = 'EQ'        # Carpeta con archivos .txt (generados por convert_knet_csv.py)
work_path2 = 'NonEQ'    # Carpeta con subcarpetas de datos NonEQ
write_path = 'result/'  # Carpeta para guardar modelo y resultados

WindowSize = 2 * SamplingRate
original_SamplingRate = 100
rate = original_SamplingRate / SamplingRate

# =========================== Procesar datos EQ ===============================
print('Cargando datos de terremotos...')
files = glob.glob(os.path.join(work_path, '*.txt'))
print(f'  Archivos EQ encontrados: {len(files)}')

X_EQ, Y_EQ, Z_EQ = [], [], []
skipped = 0
for fn in files:
    data = pd.read_csv(fn, sep=r'\s+', names=['X', 'Y', 'Z'], engine='python')
    data = data.iloc[0::int(rate)].reset_index(drop=True)

    X = data['X']
    X_peak = np.where(X == np.max(X))[0][0]

    start = X_peak - int(SamplingRate)
    end = X_peak + int(SamplingRate) * (Duration - 1)

    if start < 0:
        skipped += 1
        continue

    df = data[start:end].reset_index(drop=True)
    X_tg, Y_tg, Z_tg = df['X'], df['Y'], df['Z']

    if len(X_tg) == SamplingRate * Duration:
        for j in np.arange(0, len(X_tg) - SamplingRate, SamplingRate):
            X_batch = X_tg[j:j + WindowSize]
            Y_batch = Y_tg[j:j + WindowSize]
            Z_batch = Z_tg[j:j + WindowSize]
            if len(X_batch) == WindowSize:
                X_EQ.append(X_batch.values)
                Y_EQ.append(Y_batch.values)
                Z_EQ.append(Z_batch.values)
    else:
        skipped += 1

X_EQ = np.asarray(X_EQ)
Y_EQ = np.asarray(Y_EQ)
Z_EQ = np.asarray(Z_EQ)

n_eq_events = int(len(X_EQ) / (Duration - 1))
print(f'  Eventos EQ válidos: {n_eq_events} ({skipped} descartados)')
print(f'  Ventanas EQ: {len(X_EQ)}')

X_EQ = X_EQ.reshape(n_eq_events, Duration - 1, WindowSize, 1)
Y_EQ = Y_EQ.reshape(n_eq_events, Duration - 1, WindowSize, 1)
Z_EQ = Z_EQ.reshape(n_eq_events, Duration - 1, WindowSize, 1)

# =========================== Procesar datos NonEQ ============================
print('\nCargando datos de no-terremotos...')
X_HA, Y_HA, Z_HA = [], [], []
n_files = 0
for root, dirs, files in os.walk(work_path2):
    for folder in dirs:
        csvs = glob.glob(os.path.join(work_path2, folder, '*.csv'))
        for fn2 in csvs:
            data = pd.read_csv(fn2, header=0)
            df = data.iloc[0::int(rate)].reset_index(drop=True)

            X = df['x']
            Y = df['y']
            Z = df['z']

            X_tg = (X - np.mean(X)) / 9.80665
            Y_tg = (Y - np.mean(Y)) / 9.80665
            Z_tg = (Z - np.mean(Z)) / 9.80665

            for j in np.arange(0, len(X) - SamplingRate, SamplingRate):
                X_batch = X_tg[j:j + WindowSize]
                Y_batch = Y_tg[j:j + WindowSize]
                Z_batch = Z_tg[j:j + WindowSize]
                if len(X_batch) == WindowSize:
                    X_HA.append(X_batch.values)
                    Y_HA.append(Y_batch.values)
                    Z_HA.append(Z_batch.values)
            n_files += 1

X_HA = np.asarray(X_HA)
Y_HA = np.asarray(Y_HA)
Z_HA = np.asarray(Z_HA)

print(f'  Archivos NonEQ procesados: {n_files}')
print(f'  Ventanas NonEQ: {len(X_HA)}')

X_HA = X_HA.reshape(X_HA.shape[0], X_HA.shape[1], 1)
Y_HA = Y_HA.reshape(Y_HA.shape[0], Y_HA.shape[1], 1)
Z_HA = Z_HA.reshape(Z_HA.shape[0], Z_HA.shape[1], 1)

# =========================== Train/Test Split ================================
print('\nDividiendo datos 70/30...')
indices = np.arange(len(X_EQ))
X_EQ_train, X_EQ_test, train_idx, test_idx = train_test_split(X_EQ, indices, test_size=0.3, random_state=42)
Y_EQ_train, Y_EQ_test = Y_EQ[train_idx], Y_EQ[test_idx]
Z_EQ_train, Z_EQ_test = Z_EQ[train_idx], Z_EQ[test_idx]

X_EQ_train = X_EQ_train.reshape(X_EQ_train.shape[0] * X_EQ_train.shape[1], X_EQ_train.shape[2], 1)
Y_EQ_train = Y_EQ_train.reshape(Y_EQ_train.shape[0] * Y_EQ_train.shape[1], Y_EQ_train.shape[2], 1)
Z_EQ_train = Z_EQ_train.reshape(Z_EQ_train.shape[0] * Z_EQ_train.shape[1], Z_EQ_train.shape[2], 1)

X_EQ_test = X_EQ_test.reshape(X_EQ_test.shape[0] * X_EQ_test.shape[1], X_EQ_test.shape[2], 1)
Y_EQ_test = Y_EQ_test.reshape(Y_EQ_test.shape[0] * Y_EQ_test.shape[1], Y_EQ_test.shape[2], 1)
Z_EQ_test = Z_EQ_test.reshape(Z_EQ_test.shape[0] * Z_EQ_test.shape[1], Z_EQ_test.shape[2], 1)

indices2 = np.arange(len(X_HA))
X_HA_train, X_HA_test, train_idx2, test_idx2 = train_test_split(X_HA, indices2, test_size=0.3, random_state=42)
Y_HA_train, Y_HA_test = Y_HA[train_idx2], Y_HA[test_idx2]
Z_HA_train, Z_HA_test = Z_HA[train_idx2], Z_HA[test_idx2]

# =========================== CRNN Training ===================================
print('\n=== Entrenamiento CRNN ===')
EQ_X_train = np.dstack((X_EQ_train, Y_EQ_train, Z_EQ_train))
HA_X_train = np.dstack((X_HA_train, Y_HA_train, Z_HA_train))

EQ_y_train = np.ones(len(X_EQ_train))
HA_y_train = np.zeros(len(X_HA_train))

X_train = np.vstack((EQ_X_train, HA_X_train))
y_train = np.hstack((EQ_y_train, HA_y_train)).reshape(-1, 1)

ratio = float(len(HA_y_train) / len(EQ_y_train))
class_weights = {0: 1., 1: ratio}

print(f'  Muestras train: {len(X_train)} (EQ={len(EQ_y_train)}, NonEQ={len(HA_y_train)})')
print(f'  Class weight ratio: {ratio:.2f}')

epochs, batch_size = 100, 256
n_features = X_train.shape[2]
n_steps, n_length = 2, SamplingRate

X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))

model = tf.keras.models.Sequential([
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        input_shape=(None, n_length, n_features)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
          class_weight=class_weights, verbose=2)

model_name = f'CRNN_{SamplingRate}Hz_{Duration}s'
model.save(os.path.join(write_path, model_name + '.h5'))
print(f'\nModelo guardado: {write_path}{model_name}.h5')

# =========================== CRNN Testing ====================================
print('\n=== Evaluación CRNN ===')
EQ_X_test = np.dstack((X_EQ_test, Y_EQ_test, Z_EQ_test))
HA_X_test = np.dstack((X_HA_test, Y_HA_test, Z_HA_test))

EQ_y_test = np.ones(len(X_EQ_test))
HA_y_test = np.zeros(len(X_HA_test))

X_test = np.vstack((EQ_X_test, HA_X_test))
y_test = np.hstack((EQ_y_test, HA_y_test)).reshape(-1, 1)

X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))

print(f'  Muestras test: {len(X_test)} (EQ={len(EQ_y_test)}, NonEQ={len(HA_y_test)})')

y_prob = model.predict(X_test).ravel()

result = np.concatenate((y_test, y_prob.reshape(-1, 1)), axis=1)
df_result = pd.DataFrame(result, columns=['labels', 'prob_1'])
df_result.to_csv(os.path.join(write_path, model_name + '.csv'), index=False)
print(f'Predicciones guardadas: {write_path}{model_name}.csv')

# Quick metrics
from sklearn.metrics import roc_auc_score, classification_report
y_pred = (y_prob > 0.5).astype(int)
print(f'\nAUROC: {roc_auc_score(y_test, y_prob):.4f}')
print(classification_report(y_test.ravel(), y_pred, target_names=['NonEQ', 'EQ'], digits=4))
