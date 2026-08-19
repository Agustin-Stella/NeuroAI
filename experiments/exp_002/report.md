# Experimento exp_002 — Baseline con LOSO real (6 pacientes)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Identificación

- **ID:** exp_002
- **Fecha:** 2026-08-07
- **Autor:** Agustín Stella
- **Hipótesis:** medir la generalización REAL del baseline entre pacientes
  (Leave-One-Subject-Out), respetando el split por paciente.

## ✅ Validez metodológica

**Este SÍ es un número honesto.** A diferencia de exp_001 (split por archivo dentro
de un paciente), acá cada fold entrena en 5 pacientes y evalúa en **un paciente
nunca visto**. Estima generalización real, que es lo que importa clínicamente.

## Reproducibilidad

- **Datos:** chb01–chb06 (descarga selectiva: archivos con crisis + hasta 4 normales)
- **Montaje canónico:** 23 canales comunes a todos los archivos (harmonización
  necesaria: algunos archivos traían 24 canales)
- **Pipeline:** band-pass 0.5–40 Hz + notch 60 Hz; ventana 4 s; overlap 0
- **Features:** potencia de banda δ/θ/α/β/γ por canal → 115 features
- **Modelo:** `StandardScaler` + `LogisticRegression(class_weight="balanced")`
- **Seed:** 42

## Resultados por fold (paciente de test)

| Test | AUPRC | sens@95spec | F1 | ictales |
|---|---|---|---|---|
| chb01 | 0.721 | 0.839 | 0.361 | 112 |
| chb02 | 0.239 | 0.750 | 0.189 | 44 |
| chb03 | 0.565 | 0.621 | 0.630 | 103 |
| chb04 | 0.016 | 0.274 | 0.049 | 95 |
| chb05 | 0.721 | 0.757 | 0.075 | 140 |
| chb06 | 0.002 | 0.075 | 0.003 | 40 |

## Resumen LOSO (media ± desvío entre sujetos)

| Métrica | Valor |
|---|---|
| **AUPRC** | **0.377 ± 0.306** |
| sens@95spec | 0.553 ± 0.281 |
| ROC-AUC | 0.755 ± 0.117 |
| F1 | 0.218 ± 0.218 |

## Observaciones (las lecciones)

1. **La generalización entre pacientes es MUCHO más difícil.**
   - Dentro de un paciente (exp_001): AUPRC **0.806**.
   - Entre pacientes (exp_002): AUPRC **0.377**.
   - Se cae a menos de la mitad. Confirma en números por qué el split por paciente
     es innegociable: el número within-patient era un espejismo.

2. **Enorme variabilidad entre sujetos (±0.306).**
   - chb01/chb05: AUPRC ~0.72 (razonable).
   - chb04/chb06: AUPRC ~0.01 (falla casi total).
   - Un único split podría haber caído en chb01 y "parecer" excelente, o en chb06 y
     parecer inútil. **Por eso LOSO y no un split fijo.**

3. **chb06 y chb04 son los más desbalanceados** (0.12% y 0.47% ictal) y donde peor
   anda. El baseline lineal no captura sus patrones. Candidatos claros a analizar
   con el neurólogo y con modelos más expresivos.

## Conclusión

**AUPRC 0.377 ± 0.306 es el número honesto a batir.** Cualquier modelo futuro
(CNN 1D, etc.) tiene que superar esto de forma medible y, sobre todo, **reducir la
varianza entre sujetos**. No alcanza con mejorar la media si sigue fallando en
chb04/chb06.

## Próximos pasos

- Promover la **harmonización de canales** (montaje canónico de 23) a
  `neuropilot/preprocessing` — hoy quedó en el script del experimento.
- Etapa 3: CNN 1D, evaluada con este mismo LOSO para comparar de igual a igual.
- Preguntar al neurólogo qué distingue a chb04/chb06 (¿tipo de crisis? ¿calidad de señal?).
