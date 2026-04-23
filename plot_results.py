#!/usr/bin/env python3
"""
Genera gráficas de evaluación de los modelos CRNN y ANN.
Usa los CSV de predicciones en result/.

Uso:
    python3 plot_results.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             auc, confusion_matrix)
import os

RESULT_DIR = 'result'
P_THRESHOLD = 0.5

# Cargar resultados
crnn = pd.read_csv(os.path.join(RESULT_DIR, 'CRNN_100Hz_10s.csv'))
ann = pd.read_csv(os.path.join(RESULT_DIR, 'ANN_100Hz_10s.csv'))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('CrowdQuake — Evaluación de Modelos', fontsize=16, fontweight='bold')

# ─── 1. Curva ROC ───
ax = axes[0, 0]
for name, df, color in [('CRNN', crnn, '#2196F3'), ('ANN', ann, '#FF9800')]:
    fpr, tpr, _ = roc_curve(df['labels'], df['prob_1'])
    score = roc_auc_score(df['labels'], df['prob_1'])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUROC={score:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# ─── 2. Curva Precision-Recall ───
ax = axes[0, 1]
for name, df, color in [('CRNN', crnn, '#2196F3'), ('ANN', ann, '#FF9800')]:
    prec, rec, _ = precision_recall_curve(df['labels'], df['prob_1'])
    score = auc(rec, prec)
    ax.plot(rec, prec, color=color, lw=2.5, label=f'{name} (AUPR={score:.4f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Curva Precision-Recall')
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# ─── 3. Matrices de confusión ───
for i, (name, df, color) in enumerate([('CRNN', crnn, '#2196F3'), ('ANN', ann, '#FF9800')]):
    ax = axes[1, i]
    y_pred = (df['prob_1'] > P_THRESHOLD).astype(int)
    cm = confusion_matrix(df['labels'], y_pred)

    im = ax.imshow(cm, cmap='Blues' if i == 0 else 'Oranges', aspect='auto')
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            pct = val / cm.sum() * 100
            ax.text(c, r, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[r, c] > cm.max() * 0.5 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['NonEQ', 'EQ'])
    ax.set_yticklabels(['NonEQ', 'EQ'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusión — {name}')

plt.tight_layout()
out = os.path.join(RESULT_DIR, 'evaluacion_modelos.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Guardado: {out}')

# ─── 4. Distribución de probabilidades (figura separada) ───
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')

for i, (name, df) in enumerate([('CRNN', crnn), ('ANN', ann)]):
    ax = axes2[i]
    eq = df[df['labels'] == 1]['prob_1']
    noneq = df[df['labels'] == 0]['prob_1']

    ax.hist(noneq, bins=50, alpha=0.7, color='#4CAF50', label=f'NonEQ (n={len(noneq)})', density=True)
    ax.hist(eq, bins=50, alpha=0.7, color='#F44336', label=f'EQ (n={len(eq)})', density=True)
    ax.axvline(P_THRESHOLD, color='k', ls='--', lw=1.5, label=f'Umbral={P_THRESHOLD}')
    ax.set_xlabel('P(terremoto)')
    ax.set_ylabel('Densidad')
    ax.set_title(name)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(RESULT_DIR, 'distribucion_probabilidades.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Guardado: {out2}')

plt.show()
