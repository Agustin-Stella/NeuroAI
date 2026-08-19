# Análisis de arquitectura — NeuroPilot AI

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](LEGAL.md).


> Rol: arquitecto senior de software e investigador en IA médica.
> Base: análisis de `neuropilot-ai.md`.
> Estado: propuesta previa a implementación (sin código todavía).

---

## 1. Diagnóstico del documento

El documento es una buena visión de producto, pero como spec de arquitectura tiene tres riesgos que hay que resolver temprano, porque son caros de corregir después:

**a) El scope del MVP es demasiado ancho.** "Subir → visualizar → procesar → modelo → explicar → reportar" son 7 subsistemas. Si arrancás con los 7 en paralelo, ninguno queda sólido. En IA médica el cuello de botella nunca es el backend: es **conseguir el dato bien etiquetado y validar el modelo sin engañarte**. El 80% del valor y del riesgo está en el pipeline de datos y la evaluación.

**b) La secuencia propuesta invierte el riesgo.** El roadmap pone "Plataforma (Fase 3)" antes de tener un modelo confiable (Fase 2). Regla en ML médico: **no construyas UI sobre un modelo que todavía no sabés si generaliza.** El frontend es lo más fácil de rehacer y lo más caro de mantener si el modelo cambia de forma de salida.

**c) Falta la decisión más importante: la unidad de predicción.** El doc mezcla "detección de eventos" (encontrar inicio/fin de una crisis) con "clasificación binaria normal/epiléptico". Son problemas distintos, con datasets, etiquetas y métricas distintas. Hay que elegir uno para el MVP.

---

## 2. Decisión técnica #0 (la que condiciona todo)

**Empezar con clasificación por ventanas, no con detección de eventos.**

| | Clasificación por ventana | Detección de eventos (onset/offset) |
|---|---|---|
| Etiqueta necesaria | por segmento (ej. 4s) | timestamps precisos inicio/fin |
| Dificultad ML | baja-media (baseline en días) | alta (requiere post-procesado temporal) |
| Riesgo de leakage | manejable | alto |
| CHB-MIT lo soporta | sí, directo | sí, pero más trabajo |

El MVP debería ser: *"dada una ventana de N segundos de EEG multicanal, ¿contiene actividad ictal?"*. La localización temporal (5.4 del doc) **emerge gratis** de deslizar esa ventana sobre el registro y agrupar predicciones positivas contiguas. No necesitás un modelo de detección aparte para la primera versión.

---

## 3. Decisión técnica #1: split por paciente, no negociable

Lo elevo a **invariante de arquitectura**, no a buena práctica:

- El split train/val/test se hace **por paciente**, y se congela en un archivo versionado (`splits/v1.json`) **antes** de mirar ninguna métrica.
- CHB-MIT tiene ~24 sujetos → alcanza para un split fijo pero **no** para confiar en un solo número. La evaluación honesta acá es **Leave-One-Subject-Out (LOSO)** o k-fold por paciente. Un único split va a dar métricas con varianza enorme.
- Consecuencia arquitectónica: el pipeline de datos tiene que tener el `patient_id` como ciudadano de primera clase en cada muestra, desde la ingesta hasta la métrica.

Este es el error #1 que hunde proyectos de EEG: 98% de accuracy que se evapora en otro hospital porque el modelo memorizó al paciente, no la patología.

---

## 4. Arquitectura inicial propuesta

En vez de los 3 servicios del doc (frontend/backend/ML) desde el día uno, propongo **fases de arquitectura** que crecen con la evidencia.

### Etapa A — Research monorepo (Fases 0-2)

No hay servidor todavía. Todo es reproducibilidad.

```
neuropilot-ai/
  data/                # nunca en git — solo scripts que lo pueblan
  splits/              # splits por paciente, versionados (sí en git)
  neuropilot/          # librería Python instalable (pip install -e .)
    io/                # lectura EDF (MNE), anonimización
    preprocessing/     # filtros, artefactos, normalización
    windowing/         # segmentación + etiquetado por ventana
    datasets/          # torch Dataset, respeta el split por paciente
    models/            # baseline → CNN1D → CNN+LSTM
    training/          # loop de train, seeds, checkpoints
    evaluation/        # métricas médicas + LOSO
    explain/           # SHAP / saliency (fase 4)
  experiments/         # 1 carpeta por corrida, config + métricas + hash de git
  notebooks/           # solo exploración, nunca lógica reutilizable
```

Decisión clave: **la lógica vive en `neuropilot/` (paquete testeable), NO en los notebooks.** Los notebooks importan del paquete. Esto es lo que separa un proyecto reproducible de uno que "funcionaba en mi notebook".

### Etapa B — Servicio de inferencia (Fase 3, solo si el modelo pasa LOSO)

Recién acá aparece FastAPI, y **envuelve** la misma librería de la Etapa A. El modelo se sirve, no se reentrena en el request.

### Etapa C — Plataforma completa (Fase 4+)

Acá sí entran Postgres, Redis, Celery, React. Antes de esto, **son complejidad prematura.**

---

## 5. Decisiones técnicas concretas (con postura, no menú)

| Tema | Decisión | Por qué |
|---|---|---|
| **Formato de dato** | EDF + MNE-Python como única puerta de entrada | Estándar clínico; MNE ya resuelve montajes y unidades |
| **Ventana** | 2–4 s, con overlap solo en train (nunca en test) | Overlap en test infla métricas |
| **Muestreo** | resamplear todo a 256 Hz | CHB-MIT es 256 Hz; homogeneiza para multi-dataset futuro |
| **Baseline obligatorio** | features clásicas (banda de potencia δ/θ/α/β/γ) + regresión logística / XGBoost | Si el deep learning no le gana a esto, algo está mal. Es tu línea de honestidad |
| **Primer DL** | CNN 1D sobre la señal cruda | Menos hiperparámetros que LSTM, entrena rápido, buen techo |
| **Métrica de referencia** | Sensibilidad a especificidad fija (ej. sens@95% spec) + AUPRC | Accuracy y AUC-ROC mienten con clases desbalanceadas; el falso negativo es el costo clínico real |
| **Desbalance** | focal loss o class weights, decidido por experimento | No asumir cuál gana |
| **Tracking de experimentos** | MLflow local (o incluso carpetas + YAML al principio) | Sin esto no hay "mentalidad científica", solo corridas perdidas |
| **Config** | Hydra/YAML, cero hardcode | Reproducibilidad = config + seed + hash de git |
| **Reproducibilidad** | seed global + `torch.use_deterministic_algorithms` + log del hash de git en cada experimento | Un experimento sin estos tres datos no cuenta |

---

## 6. Riesgos que hay que mirar de frente

1. **Leakage por paciente** → resuelto con split fijo por sujeto (sección 3).
2. **Leakage por normalización** → los estadísticos de normalización se calculan **solo con train** y se aplican a val/test. Error clásico y silencioso.
3. **Overlap de ventanas cruzando el split** → una ventana no puede pisar el borde train/test.
4. **Sobreajuste al reportar** → si mirás el test set más de una vez y ajustás, ya se contaminó. El test se toca **una sola vez, al final**.
5. **Data shift entre datasets** → validación externa (entrenar en CHB-MIT, evaluar en TUH) es la prueba de fuego. Ponerla como hito, no como opcional.
6. **Ética/regulatorio** → aunque sea investigación, dejar por escrito desde ya el disclaimer de "no diagnóstico" y el manejo de datos anonimizados (usar siempre lenguaje de "asistencia/predicción", nunca "diagnóstico").

---

## 7. Roadmap de implementación (reordenado por riesgo)

Reordené las fases del doc para que **cada fase reduzca el riesgo más grande que queda vivo**.

**Fase 0 — Fundaciones de datos (2–3 semanas)**
Ingesta EDF con MNE, anonimización, y — lo más importante — el **split por paciente congelado** y el paquete `neuropilot/` con CI y tests. Entregable: un `Dataset` de PyTorch que devuelve ventanas etiquetadas respetando el split.

**Fase 1 — Baseline honesto (2–3 semanas)**
Features de banda + modelo clásico. Evaluación LOSO con métricas médicas. Entregable: **el número contra el cual todo lo demás se compara.** Si esto no supera al azar de forma robusta, el problema está en los datos, no en el modelo.

**Fase 2 — Primer deep learning (3–4 semanas)**
CNN 1D. Objetivo: **ganarle al baseline de forma medible y reproducible**, no obtener el mejor número posible. Entregable: modelo + reporte de experimento comparativo.

**Fase 3 — Validación y robustez (3–4 semanas)**
CNN+LSTM si aporta, y la prueba de fuego: **validación externa en un segundo dataset**. Entregable: evidencia de generalización (o el conocimiento honesto de que no generaliza aún).

**Fase 4 — Servicio de inferencia (3 semanas)**
FastAPI envolviendo el modelo. Endpoint: EDF → predicciones por ventana → eventos agrupados → reporte JSON. Sin base de datos todavía. Entregable: API demostrable.

**Fase 5 — Explicabilidad (paralelizable con Fase 4)**
Saliency/SHAP sobre el modelo ganador. Entregable: "qué canales y en qué momento pesaron en la decisión".

**Fase 6 — Plataforma y visor (cuando el modelo ya es confiable)**
React + visor EEG + Postgres/Redis/Celery. Recién acá el stack completo del doc tiene sentido.

La diferencia con el roadmap original: **la plataforma web pasa del medio al final**, y aparecen dos hitos nuevos que el doc no tenía — *baseline honesto* y *validación externa* — que son justamente los que separan un demo de una investigación seria.

---

## 8. Próximos pasos / decisiones a cerrar

Antes de tocar código, conviene cerrar tres decisiones, porque condicionan la estructura:

1. **¿MVP como clasificación por ventana?** (recomendación fuerte: sí).
2. **¿Empezamos solo con CHB-MIT** y dejamos TUH para validación externa? (recomiendo sí — TUH es enorme y complejo, mal primer dataset).
3. **¿Evaluación LOSO desde el inicio**, asumiendo que las métricas van a tener varianza alta y son más honestas? (recomiendo sí).

Con esas tres cerradas, el siguiente entregable natural es `docs/architecture.md` y `docs/research.md` formales, y después la estructura de carpetas de la Etapa A.
