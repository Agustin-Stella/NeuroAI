# Glosario — conceptos clave del proyecto

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Los términos que conviene manejar para entender las decisiones de NeuroPilot AI,
> explicados en una línea cada uno. Referencia rápida, no exhaustiva.
> Complementa: [`pipeline.md`](pipeline.md) (mapa de archivos) y
> [`research.md`](research.md) (metodología).

**Si tuvieras que quedarte con 5:** *LOSO*, *leakage*, *desbalance*, *normalización* y
*sensibilidad vs falsas alarmas/hora*. Con esos cinco entendés el 80 % de las decisiones.

## Señal / datos

| Término | Qué es |
|---|---|
| **EEG** | Registro de la actividad eléctrica del cerebro en el tiempo. |
| **Canal** | Cada "sensor" del EEG (un par de electrodos). Usamos 23. |
| **Montaje** | El conjunto y orden de canales. El *montaje canónico* = los 23 comunes a todos los pacientes (para que la IA reciba siempre lo mismo). |
| **Ictal / interictal** | *Ictal* = durante la crisis; *interictal* = fuera de la crisis (lo normal). |
| **Ventana (window)** | Un tramo cortito de señal (4 s) que la IA analiza como unidad. |
| **CHB-MIT** | El dataset público de EEG pediátrico con crisis anotadas que usamos. |

## Machine learning

| Término | Qué es |
|---|---|
| **Etiqueta (label)** | La "respuesta correcta" de un ejemplo (crisis=1 / normal=0). |
| **Desbalance de clases** | Solo ~1 % de las ventanas son crisis. El problema central; por eso no usamos "accuracy". |
| **CNN (red convolucional)** | El tipo de red que usamos; aprende sola qué "forma" tiene una crisis en la señal. |
| **Features** | Las características que distinguen crisis de normal. La CNN las *aprende*; el baseline las usa *hechas a mano* (potencia por bandas). |
| **Epoch** | Una pasada completa por todos los ejemplos. Entrenamos 15. |
| **Loss (pérdida)** | Cuánto se equivoca el modelo. Entrenar = bajar el loss. |
| **Overfitting (sobreajuste)** | Cuando el modelo "memoriza" en vez de aprender y falla con datos nuevos. |
| **Generalización** | Lo contrario: que funcione con un paciente nunca visto. Es *el* objetivo. |

## Hacer ciencia bien (evitar autoengañarse)

| Término | Qué es |
|---|---|
| **Leakage (fuga de datos)** | Cuando info del test se cuela en el train y las métricas mienten. El pecado #1 en EEG. |
| **Split por paciente** | Partir train/test *por persona*, nunca por ventana → evita el leakage. |
| **LOSO (Leave-One-Subject-Out)** | Entrenar dejando un paciente afuera y probar en él. Nuestra forma honesta de medir generalización. |
| **Baseline** | Un modelo simple de referencia. Sin superarlo, no podés decir que el complejo "sirve". |
| **Reproducibilidad / semilla** | Que correr lo mismo dé lo mismo. Semilla fija = azar controlado. |

## Entrenamiento (los "trucos" del desbalance)

| Término | Qué es |
|---|---|
| **Normalización / z-score** | Reescalar cada canal a media 0. Fue ~87 % de la mejora del proyecto. |
| **Undersampling (submuestreo)** | Tirar la mayoría de las ventanas normales para balancear (~15:1). |
| **`pos_weight`** | Penalizar más equivocarse en una crisis que en una normal. |
| **Checkpoint** | El modelo entrenado guardado en disco (`.pt`), para reusarlo sin reentrenar. |
| **Inferencia** | Usar el modelo ya entrenado para predecir sobre datos nuevos. |
| **Fine-tuning** | Seguir entrenando un modelo ya entrenado, con pocas epochs y LR bajo, para especializarlo sin reentrenar de cero. |
| **Adaptación al paciente** | Fine-tuning + normalizador del propio paciente para que el modelo funcione en él. Rescató chb06 (20→100 %) y chb12 (6→91 %). Ver [`adaptacion.md`](adaptacion.md). |
| **Zero-shot** | Aplicar el modelo a un paciente nuevo **sin** adaptarlo. Es el escenario difícil (falla en pacientes atípicos). |

## Evaluación médica (¿"sirve"?)

| Término | Qué es |
|---|---|
| **Sensibilidad (recall)** | De las crisis reales, cuántas detecta. La más importante (perder una crisis es lo peor). |
| **Especificidad** | De lo normal, cuánto marca bien como normal (lo contrario a falsas alarmas). |
| **AUPRC** | Métrica agregada que no se deja engañar por el desbalance (mejor que accuracy o AUC-ROC acá). |
| **sens@95spec** | Sensibilidad fijando 95 % de especificidad → número comparable entre modelos. |
| **Umbral (threshold)** | El corte de score a partir del cual se declara "crisis". Se calibra por paciente. |
| **Evento vs ventana** | Ventana = 4 s sueltos; *evento* = crisis completa (ventanas agrupadas). Se evalúa por evento, que es lo clínico. |
| **Falsas alarmas/hora** | Cuántas alarmas falsas tolerás por hora. La métrica de usabilidad real. |
| **Saliencia / explicabilidad** | Mostrar *por qué* marcó (qué canales, cuándo). Principio #1 del proyecto. |
