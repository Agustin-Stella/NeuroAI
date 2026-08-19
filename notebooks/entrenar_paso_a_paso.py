"""
ENTRENAR PASO A PASO — versión de juguete para ENTENDER, no para producción.

Problema simplificado (parecido al real pero trivial de seguir):
  - "normal" = valores bajos
  - "crisis" = valores altos
El modelo tiene que aprender solo dónde está el corte.

Corré:  .venv/bin/python notebooks/entrenar_paso_a_paso.py
Y después cambiá EPOCHS o LR abajo y volvé a correrlo para ver qué pasa.
"""
import sys
import numpy as np
import torch
import torch.nn as nn

# --- perillas que PODÉS tocar para experimentar ---
# Se pasan por línea de comando:  python ...py [EPOCHS] [LR]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30   # cuántas veces repasa TODOS los ejemplos
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1    # "tamaño del paso" al ajustar
print(f"(usando EPOCHS={EPOCHS}, LR={LR})\n")
np.random.seed(0); torch.manual_seed(0)

print("="*60)
print("1) GENERO LOS EJEMPLOS")
print("="*60)
# 100 normales (alrededor de 2) y 100 crisis (alrededor de 8)
normales = np.random.normal(2.0, 1.0, size=(100, 1))
crisis   = np.random.normal(8.0, 1.0, size=(100, 1))
X = np.vstack([normales, crisis]).astype(np.float32)
y = np.array([0]*100 + [1]*100, dtype=np.float32).reshape(-1, 1)  # 0=normal 1=crisis
X = torch.from_numpy(X); y = torch.from_numpy(y)
print(f"Tengo {len(X)} ejemplos. Normales ~2, crisis ~8.")
print("El modelo NO sabe esto: lo tiene que descubrir.\n")

print("="*60)
print("2) EL MODELO ARRANCA RANDOM (adivina mal)")
print("="*60)
# El modelo más simple posible: una recta. y = w*x + b, luego sigmoide -> probabilidad
modelo = nn.Linear(1, 1)
def prob(valor):
    with torch.no_grad():
        return torch.sigmoid(modelo(torch.tensor([[float(valor)]]))).item()
print(f"Un valor BAJO (2, debería ser ~0): el modelo dice {prob(2):.2f}")
print(f"Un valor ALTO (8, debería ser ~1): el modelo dice {prob(8):.2f}")
print("=> Adivina cualquier cosa, porque arranca sin saber nada.\n")

print("="*60)
print("3) ENTRENO: repito adivinar -> medir error -> ajustar")
print("="*60)
perdida = nn.BCEWithLogitsLoss()          # mide cuánto se equivoca
optimizador = torch.optim.SGD(modelo.parameters(), lr=LR)  # el que mueve las perillas

for epoch in range(EPOCHS):
    optimizador.zero_grad()               # limpiar ajustes previos
    logits = modelo(X)                    # (1) ADIVINAR: predice sobre todos los ejemplos
    error = perdida(logits, y)            # (2) MEDIR: qué tan mal estuvo
    error.backward()                      # (3) CALCULAR cómo mejorar cada perilla
    optimizador.step()                    # (4) AJUSTAR las perillas un poquito
    if epoch % 5 == 0 or epoch == EPOCHS-1:
        print(f"  epoch {epoch:2d}  error={error.item():.3f}")
print("=> Mirá cómo el error BAJA epoch a epoch. Eso es aprender.\n")

print("="*60)
print("4) DESPUÉS DE ENTRENAR: ahora acierta")
print("="*60)
print(f"Un valor BAJO (2): el modelo dice {prob(2):.2f}  (deberia ser ~0)")
print(f"Un valor ALTO (8): el modelo dice {prob(8):.2f}  (deberia ser ~1)")
print(f"Un valor DUDOSO (5, justo en el medio): {prob(5):.2f}")
w = modelo.weight.item(); b = modelo.bias.item()
corte = -b / w if w != 0 else float('nan')
print(f"\nEl modelo aprendió que el corte normal/crisis esta cerca de {corte:.1f}")
print("(¡y nadie se lo dijo, lo dedujo de los ejemplos!)")
