#!/bin/bash
# Pipeline completo de entrenamiento CrowdQuake
# Uso: bash run_training.sh
set -e

cd "$(dirname "$0")"

echo "============================================"
echo " CrowdQuake - Pipeline de Entrenamiento"
echo "============================================"
echo ""

# Paso 1: Convertir K-NET CSV → .txt
echo "[1/3] Convirtiendo datos K-NET CSV → .txt ..."
python3 convert_knet_csv.py --threshold 0.05
echo ""

# Verificar que hay datos
N_EQ=$(ls EQ/*.txt 2>/dev/null | wc -l)
N_NONEQ=$(find NonEQ -name '*.csv' 2>/dev/null | wc -l)
echo "Datos disponibles: $N_EQ archivos EQ, $N_NONEQ archivos NonEQ"

if [ "$N_EQ" -eq 0 ]; then
    echo "ERROR: No se generaron archivos .txt en EQ/"
    exit 1
fi
echo ""

# Paso 2: Entrenar CRNN
echo "[2/3] Entrenando CRNN..."
python3 CRNN.py
echo ""

# Paso 3: Entrenar ANN
echo "[3/3] Entrenando ANN..."
python3 ANN.py
echo ""

echo "============================================"
echo " Pipeline completado"
echo " Resultados en: result/"
echo "============================================"
ls -lh result/
