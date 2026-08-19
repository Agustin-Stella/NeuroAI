# Arquitectura — NeuroPilot AI

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Documento de arquitectura técnica. Congela las decisiones acordadas.
> Complementa: `neuropilot-ai.md` (visión de producto) y `docs/research.md` (metodología científica).
> Estado: aprobado para Etapa A (research monorepo).

---

## 1. Principio rector

La arquitectura **crece con la evidencia**, no de golpe. No se construye plataforma web sobre un modelo que todavía no sabemos si generaliza. El orden es: **dato confiable → baseline honesto → modelo → validación externa → servicio → plataforma**.

Tres invariantes que no se negocian:

1. **Split por paciente**, congelado y versionado, antes de mirar ninguna métrica.
2. **La lógica vive en el paquete `neuropilot/`**, testeable. Los notebooks solo exploran.
3. **Cada experimento es reproducible**: config + seed + hash de git + dataset versionado.

---

## 2. Decisión de modelado: clasificación por ventana

El MVP **no** detecta "la crisis completa". Clasifica ventanas cortas y agrupa.

```
Input:   ventana EEG de 4 s (multicanal)
Output:  ictal probability ∈ [0, 1]

ventana 1: 0.02  → normal
ventana 2: 0.05  → normal
ventana 3: 0.93  → ictal
ventana 4: 0.95  → ictal
ventana 5: 0.91  → ictal
                 ↓ agrupación de positivas contiguas
Evento:  12:43:10 – 12:43:22
```

La localización temporal **emerge** de deslizar la ventana; no requiere un modelo de detección aparte.

---

## 3. Etapas de arquitectura

### Etapa A — Research monorepo (Etapas 0–3 del roadmap)
Sin servidor. Todo es reproducibilidad y método. Es donde vivimos ahora.

### Etapa B — Servicio de inferencia (Etapa 5, solo si el modelo pasa validación externa)
FastAPI **envuelve** el mismo paquete `neuropilot/`. El modelo se sirve, no se reentrena en el request.
Endpoint conceptual: `EDF → predicciones por ventana → eventos agrupados → reporte JSON`.

### Etapa C — Plataforma completa (Etapa 5+)
React + visor EEG + Postgres + Redis + Celery. Antes de tener un modelo confiable, son **complejidad prematura**.

---

## 4. Estructura del repositorio

```
neuropilot-ai/
├── neuropilot/              # librería instalable (pip install -e .)
│   ├── data/
│   │   ├── loaders.py       # lectura EDF (MNE), anonimización
│   │   └── splits.py        # split por paciente, congelado y versionado
│   ├── preprocessing/
│   │   ├── filters.py       # band-pass, notch
│   │   └── normalization.py # stats calculados SOLO con train
│   ├── windowing/
│   │   └── segment.py       # ventaneo + etiquetado por ventana
│   ├── datasets/
│   │   └── eeg_dataset.py   # torch Dataset, respeta el split
│   ├── models/
│   │   ├── baseline.py      # features de banda + modelo clásico
│   │   └── cnn1d.py         # CNN 1D sobre señal cruda
│   ├── training/
│   │   └── trainer.py       # loop, seeds, checkpoints, logging MLflow
│   └── evaluation/
│       └── metrics.py       # métricas médicas + LOSO
│
├── experiments/             # 1 carpeta por corrida (config + métricas + hash git)
├── notebooks/               # SOLO exploración, nunca lógica reutilizable
├── configs/                 # YAML por experimento, cero hardcode
├── tests/
├── docs/                    # vision / architecture / research
├── data/                    # gestionado por DVC, nunca en git
├── splits/                  # splits por paciente, versionados (sí en git)
└── README.md
```

**Regla dura:** los notebooks importan de `neuropilot/`. Si un notebook tiene lógica que querés reusar, esa lógica se muda al paquete.

---

## 5. Flujo de datos (Etapa A)

```
EDF (CHB-MIT)
    ↓  data/loaders.py      — carga con MNE, anonimiza, resamplea a 256 Hz
    ↓  preprocessing/        — band-pass + notch, normalización (stats de train)
    ↓  windowing/segment.py  — ventanas de 4 s, overlap solo en train
    ↓  datasets/eeg_dataset  — (tensor, label, patient_id)
    ↓  models/               — baseline → cnn1d
    ↓  training/trainer.py   — entrena, loguea a MLflow
    ↓  evaluation/metrics.py — sens@spec, AUPRC, LOSO
    ↓  experiments/exp_NNN/  — reporte reproducible
```

`patient_id` viaja con cada muestra de punta a punta: es lo que hace posible el split honesto y la evaluación LOSO.

---

## 6. Decisiones técnicas (con postura)

| Tema | Decisión | Por qué |
|---|---|---|
| Formato de entrada | EDF + MNE-Python, única puerta | estándar clínico; MNE resuelve montajes/unidades |
| Muestreo | resamplear todo a 256 Hz | homogeneiza para multi-dataset futuro |
| Ventana | 4 s; overlap **solo** en train | overlap en test infla métricas |
| Baseline obligatorio | features de banda (δ/θ/α/β/γ) + LogReg/XGBoost | línea de honestidad: el DL tiene que ganarle |
| Primer DL | CNN 1D sobre señal cruda | pocos hiperparámetros, entrena rápido, buen techo |
| Rama alternativa | representación tiempo-frecuencia (espectrograma) + CNN 2D | muy usada en EEG; señal → "imagen" |
| Métrica de referencia | sensibilidad @ especificidad fija + AUPRC | accuracy/AUC-ROC mienten con desbalance; el falso negativo es el costo clínico |
| Desbalance | focal loss o class weights, decidido por experimento | no asumir cuál gana |
| Evaluación | LOSO (Leave-One-Subject-Out) desde el inicio | con ~24 sujetos, un solo split tiene varianza enorme |
| Config | YAML por experimento, cero hardcode | reproducibilidad = config + seed + hash git |
| Reproducibilidad | seed global + algoritmos deterministas + hash de git logueado | experimento sin esos tres datos no cuenta |

---

## 7. MLOps (liviano, desde temprano)

Sin Kubernetes ni nada exagerado. Solo lo que evita perder trabajo:

- **DVC** para versionar datos. Vas a tener `dataset_v1`, `dataset_filtrado`, `dataset_final` — sin DVC no sabés cuál usaste en el experimento #34.
- **MLflow** para experimentos. Cada corrida registra modelo, dataset, ventana, hiperparámetros y métricas. Es lo que convierte "corridas perdidas" en un historial comparable.

```
Experimento #34
  modelo:    CNN1D
  dataset:   CHB-MIT (dvc: a3f1c…)
  ventana:   4 s
  lr:        0.001
  F1:        0.87
  recall:    0.91
```

---

## 8. Modelos: hoja de ruta

```
Baseline           features de banda → LogReg/XGBoost
   ↓
CNN 1D             señal cruda → conv → clasificador
   ↓
Time-frequency     espectrograma → CNN 2D
   ↓
Transformer        atención sobre la secuencia → predicción + interpretabilidad
```

Cada escalón debe **ganarle de forma medible** al anterior. Si no, no se adopta.

---

## 9. Lo que esta arquitectura deja explícitamente para después

- Base de datos, colas (Celery), cache (Redis): recién en Etapa C.
- Visor EEG en React: Etapa C.
- Asistente conversacional / LLM: fuera del MVP.
- Multimodalidad (MRI, historia clínica): visión a largo plazo, no ahora.

El objetivo de esta arquitectura es **no gastar complejidad antes de tener un modelo que la justifique**.
