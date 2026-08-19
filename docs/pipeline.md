# Pipeline de IA — mapa de archivos

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Recorrido de los módulos que van de la señal EEG cruda al modelo entrenado y a la
> detección. Ordenado por el **flujo de datos**. Sirve como puerta de entrada al código.
> Complementa: [`architecture.md`](architecture.md) (decisiones de arquitectura) y
> [`research.md`](research.md) (metodología científica).

Marcados con ⭐ los tres del corazón de "cómo aprende": normalización, la red y el
entrenador.

## Orden de ejecución al entrenar

```
loaders → channels → filters → segment → normalization → dataset → cnn1d → trainer
                                    ⭐ (normaliza)             ⭐ (la red) ⭐ (aprende)
```

---

## Cómo pasa el dato de un paso a otro

Pensá el dato como algo en una cinta transportadora: cada módulo lo recibe en una forma
y lo entrega transformado al siguiente. Qué entra y qué sale, en criollo:

| Paso | El dato **entra** como… | …y **sale** como… |
|---|---|---|
| `loaders` | un archivo `.edf` | una matriz de **23 canales × ~920.000 números** (1 hora de señal, en voltios) + la lista de crisis (en segundos) |
| `channels` | esa matriz | la misma, recortada a los **23 canales canónicos** en orden fijo |
| `filters` | señal cruda | la misma señal, **limpia** (sin ruido de línea ni deriva) |
| `segment` | 1 hora de señal continua | **~900 ventanas** de 4 s (cada una `23 × 1024` números) + **una etiqueta 0/1 por ventana** |
| `normalization` ⭐ | ventanas en voltios (números chiquitos, distinta escala por paciente) | las mismas ventanas **centradas** (media 0, desvío 1) → comparables entre pacientes |
| `dataset` | el montón de ventanas | pares **(ventana, etiqueta)** entregados de a uno al modelo |
| `cnn1d` ⭐ | una ventana `23 × 1024` | **un solo número: el score de crisis** (0 a 1) |
| `trainer` ⭐ (entrenando) | el score + la etiqueta correcta | **un ajuste a la red** (así aprende); repetido miles de veces |
| `trainer` (usando) | ventanas nuevas | **un score por ventana**, sin ajustar nada |
| `calibration` | todos los scores del paciente | **un umbral de alarma** ajustado a ese paciente |
| `events` | las ventanas con score alto | **eventos** con inicio y fin en segundos ("crisis en el min 50") |
| `detector` | el EDF entero | la **lista de eventos + gráfico + explicación** |

**En una frase:** una hora de señal se corta en ~900 ventanitas, cada ventanita se
convierte en un número (probabilidad de crisis), esos números se calibran y se agrupan,
y lo que sale son los rangos de tiempo donde hay que mirar.

La forma del dato, resumida:

```
.edf  →  (23 × 920000)  →  (900 × 23 × 1024)  →  900 scores  →  eventos (min 50, min 52…)
archivo    señal 1 hora        ventanas de 4s      un nº c/u        rangos de tiempo
```

---

## 1. Cargar los datos

| Archivo | Qué hace |
|---|---|
| [`data/loaders.py`](../neuropilot/data/loaders.py) | Lee los `.edf` con MNE y parsea el `summary.txt` de cada paciente (frecuencia, canales y **cuándo ocurre cada crisis**). Solo lee: no toca la señal. |
| [`data/splits.py`](../neuropilot/data/splits.py) | Divide **por paciente** (nunca mezcla ventanas de un paciente entre train y test — sería leakage). Split congelado + folds **LOSO** (dejar-un-paciente-afuera). |

## 2. Preprocesamiento (acá vive la normalización)

| Archivo | Qué hace |
|---|---|
| [`preprocessing/filters.py`](../neuropilot/preprocessing/filters.py) | Limpia la señal: pasa-banda 0.5–40 Hz (deriva y ruido alto) + notch 60 Hz (línea eléctrica). |
| ⭐ [`preprocessing/normalization.py`](../neuropilot/preprocessing/normalization.py) | `ChannelNormalizer`: z-score por canal `(x − media)/desvío`. **Invariante anti-leakage: media/desvío se calculan solo con train.** Fue ~87 % de la mejora del proyecto (exp_004→exp_006): sin esto, lo aprendido en un paciente no se traslada a otro. |
| [`preprocessing/channels.py`](../neuropilot/preprocessing/channels.py) | Montaje canónico: los 23 canales comunes a todos los pacientes, mismo orden (la CNN necesita entrada de tamaño fijo). Es lo que rompió con chb12. |

## 3. Cortar en ejemplos

| Archivo | Qué hace |
|---|---|
| [`windowing/segment.py`](../neuropilot/windowing/segment.py) | Corta la señal continua en **ventanas de 4 s** y las etiqueta (1=crisis / 0=normal) según las anotaciones. Convierte 1 h de EEG en ~900 ejemplos. |
| [`datasets/eeg_dataset.py`](../neuropilot/datasets/eeg_dataset.py) | El "servidor de ejemplos" para PyTorch: junta loaders → filtros → ventaneo → normalización y entrega `(ventana, etiqueta)`. Memory-bounded (no carga todo a RAM). |

## 4. Los modelos

| Archivo | Qué hace |
|---|---|
| ⭐ [`models/cnn1d.py`](../neuropilot/models/cnn1d.py) | **La red neuronal.** CNN 1D (Conv→BatchNorm→ReLU→MaxPool) que aprende sus propias características de la señal cruda y da un score de crisis por ventana. Es el modelo que corre en la web. |
| [`models/baseline.py`](../neuropilot/models/baseline.py) | Modelo clásico de referencia (regresión logística sobre potencia por bandas). Sirve para comparar honestamente si la CNN supera a algo simple. |

## 5. El entrenamiento

| Archivo | Qué hace |
|---|---|
| ⭐ [`training/trainer.py`](../neuropilot/training/trainer.py) | **El motor de aprendizaje.** Bucle *adivinar→medir error→ajustar* (`train_model`), manejo del desbalance con `pos_weight`, semillas para reproducibilidad, y `predict_proba`. Genérico para cualquier modelo PyTorch. |
| [`training/run_loso.py`](../neuropilot/training/run_loso.py) | **Entrenador de evaluación (LOSO).** Entrena N modelos (uno por paciente dejado afuera) para medir generalización honesta. Reanudable, con cache de ventanas y guardado por fold. |
| [`training/train_deployment.py`](../neuropilot/training/train_deployment.py) | **Entrenador del modelo de producción.** Un solo modelo con TODOS los pacientes, checkpoint autocontenido → `models/deployment_v1.pt`, el que usa la web. |

**Dos entrenamientos a propósito:** LOSO mide *cuán bien va a andar con un paciente nuevo*
(81 % de crisis, 8/9 pacientes al 100 %); el modelo de despliegue es el que *se usa*.

## 6. Evaluación y calibración

| Archivo | Qué hace |
|---|---|
| [`evaluation/metrics.py`](../neuropilot/evaluation/metrics.py) | Métricas médicas (sensibilidad, especificidad, AUPRC, sens@95spec). Nada de solo "accuracy", que miente con desbalance. |
| [`evaluation/calibration.py`](../neuropilot/evaluation/calibration.py) | Ajusta el **umbral de alarma por paciente** sin etiquetas (percentil de los scores). Hace usable el modelo con un paciente nuevo. |
| [`evaluation/events.py`](../neuropilot/evaluation/events.py) | Convierte ventanas sueltas en **eventos** (rangos temporales) y mide viabilidad clínica (crisis detectadas, falsas alarmas/hora). |

## 7. Uso y explicación

| Archivo | Qué hace |
|---|---|
| [`inference/detector.py`](../neuropilot/inference/detector.py) | MVP end-to-end: EDF → detección. Ata todo el pipeline; es lo que llama la web ([`app/`](../app/)). |
| [`explain/saliency.py`](../neuropilot/explain/saliency.py) | Explicabilidad: qué canales y en qué momento activaron la detección (saliencia por gradiente). |

---

## El desbalance, en una nota

Solo ~1 % de las ventanas son crisis. Sin cuidado, un modelo que dice "normal" siempre
acierta el 99 % y es inútil. Se ataca en dos lados: **submuestreo** de negativos (~15:1,
en `run_loso.py` / `train_deployment.py`) y **`pos_weight`** en la pérdida (en `trainer.py`).
Por eso también se reporta AUPRC/sensibilidad, no accuracy.
