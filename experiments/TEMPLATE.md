# Experimento exp_NNN

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Copiar esta plantilla a `experiments/exp_NNN/report.md` por cada corrida.
> Un experimento sin config + seed + hash de git + dataset versionado **no cuenta**.

## Identificación

- **ID:** exp_NNN
- **Fecha:**
- **Autor:**
- **Hipótesis:** (¿qué esperás demostrar con esta corrida?)

## Reproducibilidad

- **Config:** `configs/....yaml`
- **Seed:**
- **Git hash:**
- **Dataset (DVC):**
- **MLflow run id:**

## Setup

- **Modelo:**
- **Ventana:** 4 s
- **Split:** LOSO / por paciente
- **Preprocessing:** band-pass, notch, normalización (stats de train)

## Resultados

| Métrica | Valor (media ± desvío entre sujetos) |
|---|---|
| Sensibilidad @ 95% spec | |
| Sensibilidad / Recall | |
| Especificidad | |
| Precision | |
| F1 | |
| AUPRC | |
| FNR | |
| FPR | |

## Comparación

- **Contra qué se compara:** (baseline / experimento anterior)
- **¿Ganó de forma medible?** sí / no

## Observaciones

- Variabilidad entre pacientes:
- Casos donde falla:
- Próximo paso:

## Comunicación (lenguaje responsable)

> El modelo identifica patrones EEG compatibles con actividad epiléptica dentro del dataset evaluado.
> (Nunca "diagnóstico confirmado".)
