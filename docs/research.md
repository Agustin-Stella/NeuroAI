# Metodología científica — NeuroPilot AI

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Cómo hacemos ciencia en este proyecto para no engañarnos.
> Complementa: `neuropilot-ai.md` (visión) y `docs/architecture.md` (arquitectura técnica).

---

## 1. Pregunta de investigación (MVP)

> Dada una ventana de 4 segundos de EEG multicanal, ¿podemos estimar la probabilidad de que contenga actividad **ictal**, de forma que **generalice a pacientes no vistos** y — idealmente — a otro dataset?

No es "detectar epilepsia". Es estimar probabilidad ictal por ventana, con honestidad estadística.

---

## 2. Fase -1: entender el dominio (antes de tocar IA)

Si no entendés qué predecís, entrenás una caja negra. Antes del primer modelo, dejar por escrito (en `docs/` o en el notebook de exploración):

- Qué es un EEG y cómo se registra (electrodos, montaje, canales).
- Crisis **focal** vs **generalizada**.
- **Ictal / interictal / preictal / postictal**: qué significa cada estado.
- Cómo se **etiqueta** un EEG y quién lo hace.
- Qué son artefactos (parpadeo, movimiento, línea de 50/60 Hz) y por qué contaminan.

Entregable: media página de glosario propio. Si no lo podés explicar con tus palabras, todavía no estás listo para modelar.

---

## 3. La parte humana (no la saltamos)

Un modelo con 98% de accuracy que nadie usa no vale nada. Antes de invertir meses, hablar con **al menos un usuario real** — un neurólogo, un estudiante de medicina o un investigador — para validar el dolor.

Pregunta abierta clave: *¿el mayor problema es realmente detectar crisis, o es otra cosa?* Hipótesis a testear con usuarios:

- "Me lleva 40 minutos escribir el informe." → el valor podría estar en el **reporte**, no en la detección.
- "Necesito comparar este EEG con uno de hace 6 meses." → el valor podría estar en la **comparación histórica**.

Ese insight puede cambiar la prioridad del roadmap. Documentar lo que se aprenda.

---

## 4. Invariantes metodológicas (errores que hunden proyectos de EEG)

1. **Split por paciente.** Ninguna ventana de un paciente de test aparece en train. Se congela en `splits/` antes de mirar métricas.
2. **Normalización sin leakage.** Media/desvío se calculan **solo con train** y se aplican a val/test.
3. **Overlap solo en train.** En test, ventanas sin solapamiento; el overlap infla artificialmente los números.
4. **El test se toca una sola vez.** Si lo mirás y ajustás, se contaminó. Todo el desarrollo se hace contra validación.
5. **Baseline primero.** Sin una línea clásica de comparación, no sabés si tu deep learning aporta algo.

---

## 5. Protocolo de evaluación

- **LOSO (Leave-One-Subject-Out)** como evaluación principal. Con ~24 sujetos en CHB-MIT, un único split da varianza enorme; reportamos media ± desvío entre sujetos.
- **Métricas** (nunca accuracy sola, por el desbalance de clases):
  - Sensibilidad / Recall (prioritaria: el falso negativo es el costo clínico).
  - Especificidad.
  - Precision, F1.
  - AUPRC (más informativa que AUC-ROC con clases desbalanceadas).
  - Sensibilidad a especificidad fija (ej. sens @ 95% spec) como número comparable entre modelos.
  - False Positive Rate y False Negative Rate.
- **Reportar variabilidad entre pacientes**, no solo el promedio. Un modelo que va excelente en 20 sujetos y falla en 4 no es "bueno en promedio": es inestable.

### Por qué "accuracy" engaña — con nuestros números

Como solo ~1 % de las ventanas son ictales, un modelo que diga *"normal"* siempre tendría
~99 % de accuracy **sin detectar ni una crisis**. Ese número está prohibido acá. En su
lugar reportamos **cobertura por evento a una tasa de falsas alarmas fija**:

- Cobertura real (LOSO, pacientes no vistos): **81 % de las crisis detectadas** (39/48).
- **8 de 9 pacientes: 100 %** de sus crisis. El que falla (chb06) es una limitación de
  datos —crisis cortas y sin contraste—, documentada en el error analysis.
- Operando a **< 1 falsa alarma por hora** se mantiene ~77 % de cobertura.

La regla: se reporta la **sensibilidad por evento a especificidad / FA-por-hora fija**,
nunca el accuracy suelto.

### Reality-check: paciente 100 % no visto (chb12)

El 81 % sale del LOSO sobre chb01–09. Como test más duro, se corrió el modelo de
despliegue sobre **chb12**, excluido por completo del entrenamiento: **detectó 1 de 14
crisis**. Lección metodológica: **el promedio LOSO puede ser optimista** para pacientes
con morfología de crisis atípica (cortas/sutiles, como chb06 y chb12). La generalización
*zero-shot* a un paciente arbitrario es un problema abierto; por eso el reporte honesto
distingue "promedio sobre el pool" de "un paciente fresco puede fallar feo", y por eso el
próximo paso es **adaptación al paciente** y/o **más datos diversos** (ver `historia.md`).

---

## 6. Validación externa (lo que separa demo de investigación)

- Entrenar en **CHB-MIT**, evaluar en un **dataset externo** (ej. TUH EEG Corpus).
- Se **espera degradación** por *domain shift* (otro hospital, otro equipo, otro protocolo). Reportarla con honestidad es el resultado científico, no un fracaso.
- Formulación correcta de resultados:
  - ❌ "Mi modelo tiene 98% de accuracy."
  - ✅ "Entrené en CHB-MIT y evalué en un dataset externo, con degradación esperable de X a Y en sensibilidad@95%spec."

---

## 7. Reproducibilidad — checklist por experimento

Un experimento **no cuenta** si le falta alguno de estos:

- [ ] Config en `configs/` (YAML), sin valores hardcodeados.
- [ ] Seed global fijado (numpy, torch, python) + algoritmos deterministas.
- [ ] Hash de git del código que corrió.
- [ ] Versión del dataset (DVC).
- [ ] Métricas registradas en MLflow.
- [ ] Reporte en `experiments/exp_NNN/` (usar `experiments/TEMPLATE.md`).

---

## 8. Comunicación de resultados (lenguaje médico responsable)

Nunca presentar como diagnóstico. Lenguaje médico responsable:

- ❌ "El modelo detecta epilepsia."
- ✅ "El modelo identifica patrones EEG compatibles con actividad epiléptica dentro del dataset evaluado."

Términos permitidos: *asistencia, predicción, detección, patrones compatibles*. Prohibido: *diagnóstico confirmado*.

---

## 9. Datasets

| Dataset | Uso | Notas |
|---|---|---|
| **CHB-MIT Scalp EEG** | entrenamiento + evaluación LOSO | ~24 sujetos, 256 Hz, EDF, anotaciones de crisis |
| **TUH EEG Corpus** | validación externa | grande y heterogéneo; mal primer dataset, buen segundo |

Todo dato se maneja **anonimizado**. CHB-MIT ya lo está.
