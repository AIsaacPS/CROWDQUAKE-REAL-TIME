#!/usr/bin/env python3
"""
====================================
Implementation of ANN (sklearn compatible)
====================================
Basado en el código original de CrowdQuake (KDD 2020).
Fix: sklearn.externals.joblib → joblib directo.
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# =========================== Configuración ===================================
SamplingRate = 100       # 25, 50, o 100 Hz
Duration = 10            # 2, 4, o 10 segundos

work_path = 'EQ'
work_path2 = 'NonEQ'
write_path = 'result/'

WindowSize = 2 * SamplingRate
original_SamplingRate = 100
rate = original_SamplingRate / SamplingRate

# =========================== Procesar datos EQ ===============================
print('Cargando datos EQ y extrayendo features...')
files = glob.glob(os.path.join(work_path, '*.txt'))
print(f'  Archivos EQ: {len(files)}')

EQ_features = []
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
        VS = pow(pow(X_tg, 2) + pow(Y_tg, 2) + pow(Z_tg, 2), 1/2)
        for j in np.arange(0, len(X_tg) - SamplingRate, SamplingRate):
            train = VS[j:j + WindowSize]
            if len(train) == WindowSize:
                Q75, Q25 = np.percentile(train, [75, 25])
                IQR = Q75 - Q25
                ZCx = ZCy = ZCz = 0
                CAV = 0
                for i in range(j, j + WindowSize - 1):
                    if (X_tg[i] < 0) != (X_tg[i+1] < 0): ZCx += 1
                    if (Y_tg[i] < 0) != (Y_tg[i+1] < 0): ZCy += 1
                    if (Z_tg[i] < 0) != (Z_tg[i+1] < 0): ZCz += 1
                    CAV += (VS[i] + VS[i+1]) / 2.0 / SamplingRate
                EQ_features.append([IQR, max(ZCx, ZCy, ZCz), CAV])
    else:
        skipped += 1

EQ_features = np.reshape(EQ_features, (int(len(EQ_features) / (Duration - 1)), Duration - 1, 3))
print(f'  Eventos EQ: {len(EQ_features)} ({skipped} descartados)')

# =========================== Procesar datos NonEQ ============================
print('\nCargando datos NonEQ y extrayendo features...')
HA_features = []
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

            VS = pow(pow(X_tg, 2) + pow(Y_tg, 2) + pow(Z_tg, 2), 1/2)

            for j in np.arange(0, len(X) - SamplingRate, SamplingRate):
                train = VS[j:j + WindowSize]
                if len(train) == WindowSize:
                    Q75, Q25 = np.percentile(train, [75, 25])
                    IQR = Q75 - Q25
                    ZCx = ZCy = ZCz = 0
                    CAV = 0
                    for i in range(j, j + WindowSize - 1):
                        if (X_tg[i] < 0) != (X_tg[i+1] < 0): ZCx += 1
                        if (Y_tg[i] < 0) != (Y_tg[i+1] < 0): ZCy += 1
                        if (Z_tg[i] < 0) != (Z_tg[i+1] < 0): ZCz += 1
                        CAV += (VS[i] + VS[i+1]) / 2.0 / SamplingRate
                    HA_features.append([IQR, max(ZCx, ZCy, ZCz), CAV])
            n_files += 1

HA_features = np.reshape(HA_features, (len(HA_features), 1, 3))
print(f'  Archivos NonEQ: {n_files}, Ventanas: {len(HA_features)}')

# =========================== Train/Test Split ================================
EQ_train, EQ_test = train_test_split(EQ_features, test_size=0.3, random_state=42)
HA_train, HA_test = train_test_split(HA_features, test_size=0.3, random_state=42)

# =========================== K-Means Balancing ===============================
EQ_train = np.reshape(EQ_train, (len(EQ_train) * (Duration - 1), 3))
EQ_train_y = np.ones(len(EQ_train))
HA_train = np.reshape(HA_train, (len(HA_train), 3))

if len(EQ_train) < len(HA_train):
    print(f'\nBalanceando con K-Means (EQ={len(EQ_train)} < NonEQ={len(HA_train)})...')
    kmeans = KMeans(n_clusters=len(EQ_train), random_state=42, n_init=10).fit(HA_train)
    HA_train_centroid = kmeans.cluster_centers_
    HA_train_y = np.zeros(len(HA_train_centroid))
    ANN_train_X = np.vstack((EQ_train, HA_train_centroid))
    ANN_train_y = np.hstack((EQ_train_y, HA_train_y))
else:
    HA_train_y = np.zeros(len(HA_train))
    ANN_train_X = np.vstack((EQ_train, HA_train))
    ANN_train_y = np.hstack((EQ_train_y, HA_train_y))

EQ_test = np.reshape(EQ_test, (len(EQ_test) * (Duration - 1), 3))
EQ_test_y = np.ones(len(EQ_test))
HA_test = np.reshape(HA_test, (len(HA_test), 3))
HA_test_y = np.zeros(len(HA_test))
ANN_test_X = np.vstack((EQ_test, HA_test))
ANN_test_y = np.hstack((EQ_test_y, HA_test_y))

# =========================== ANN Training ====================================
print(f'\n=== Entrenamiento ANN ===')
print(f'  Train: {len(ANN_train_X)} muestras')

min_max_scaler = preprocessing.MinMaxScaler()
ANN_train_X = min_max_scaler.fit_transform(ANN_train_X)
ANN_test_X = min_max_scaler.transform(ANN_test_X)

model_name = f'ANN_{SamplingRate}Hz_{Duration}s'
joblib.dump(min_max_scaler, os.path.join(write_path, model_name.replace('ANN', 'ANN_scaler') + '.pkl'))

mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='sgd',
                    alpha=0, max_iter=10000, random_state=42, learning_rate_init=0.2)
mlp.fit(ANN_train_X, ANN_train_y.ravel())
joblib.dump(mlp, os.path.join(write_path, model_name + '.pkl'))

# =========================== ANN Testing =====================================
print(f'\n=== Evaluación ANN ===')
print(f'  Test: {len(ANN_test_X)} muestras')

y_prob = mlp.predict_proba(ANN_test_X)

result = np.concatenate((ANN_test_y.reshape(-1, 1), y_prob), axis=1)
df_result = pd.DataFrame(result, columns=['labels', 'prob_0', 'prob_1'])
df_result.to_csv(os.path.join(write_path, model_name + '.csv'), index=False)

from sklearn.metrics import roc_auc_score, classification_report
y_pred = (y_prob[:, 1] > 0.5).astype(int)
print(f'\nAUROC: {roc_auc_score(ANN_test_y, y_prob[:, 1]):.4f}')
print(classification_report(ANN_test_y, y_pred, target_names=['NonEQ', 'EQ'], digits=4))
print(f'Modelo guardado: {write_path}{model_name}.pkl')
print(f'Predicciones guardadas: {write_path}{model_name}.csv')
