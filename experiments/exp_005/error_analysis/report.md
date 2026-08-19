# Error analysis — por qué chb06 colapsa (AUPRC 0.002) y chb09 no (0.91)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../../LEGAL.md).


Modelos recargados desde los checkpoints de exp_005 (sin reentrenar). Figura:
[scores_chb06_vs_chb09.png](scores_chb06_vs_chb09.png).

## 1) ¿FN en crisis o FP en interictal? → **Falsos negativos: no "ve" las crisis**

| | chb06 (colapsa) | chb09 (funciona) |
|---|---|---|
| Sens @ umbral 0.5 | **0.03** (1 de 40 crisis) | 0.99 (68 de 69) |
| TP / FN @0.5 | 1 / **39** | 68 / 1 |
| Score ictal (mediana) | **0.010** | 1.000 |
| Score interictal (mediana) | 0.001 | 0.969 |

En chb06 los scores de las ventanas de crisis quedan **pegados a cero**, casi
indistinguibles del interictal. El error dominante NO es falsas alarmas en el
interictal: es que **el modelo no le asigna score alto a las crisis de chb06**.
Hay una señal de ranking mínima (68% de las crisis por encima de la mediana
interictal → ROC-AUC 0.64), pero insuficiente para cualquier umbral útil.

## 2) ¿Qué tiene de distinto chb06? → **crisis cortas y sin contraste de amplitud**

| | chb06 | chb09 |
|---|---|---|
| sfreq / canales orig. | 256 Hz / 23 | 256 Hz / 23 → **idénticos** |
| Nº de crisis | 10 | 4 |
| Duración crisis (mediana) | **15 s** (12–20) | **68 s** (62–79) |
| Ventanas ictales | 40 | 69 |
| positive_rate | **0.117 %** (1/852) | 0.518 % (1/192) |
| RMS ictal vs interictal | **70 vs 82 µV** (sin contraste) | **245 vs 45 µV** (5.4×) |

- **No es montaje/canales ni frecuencia**: son idénticos (23 canales, 256 Hz).
- **No es "más artefactos"**: la amplitud interictal es similar entre pacientes.
- **Sí es la naturaleza de las crisis**: las de chb06 son **cortas** (15 s → muy pocas
  ventanas ictales) y **electrográficamente sutiles** — no sobresalen en amplitud
  (70 vs 82 µV). Las de chb09 son largas y dramáticas (5.4× la amplitud interictal),
  triviales de detectar. La fila inferior de la figura lo muestra: en chb09 las
  distribuciones de RMS ictal/interictal están separadas; en chb06 se superponen.

## 3) Veredicto: **problema de datos/señal, no de capacidad del modelo**

Tres evidencias convergen:

1. El modelo aprendió, de sujetos como chb09, que "crisis = evento de gran amplitud".
   Las crisis de chb06 no tienen esa firma → quedan invisibles. Es un problema de
   **distribución/morfología**, no de que falte modelar la dinámica temporal.
2. **El baseline (bandpower + logreg) también dio AUPRC 0.002 en chb06** (exp_004).
   Dos familias de modelo completamente distintas fallan igual → el límite está en
   **los datos de chb06**, no en la arquitectura.
3. Con solo 40 ventanas ictales cortas y sin contraste, no hay señal suficiente para
   que ninguna capacidad extra "invente" separabilidad.

**Conclusión:** agregar CNN+LSTM **no rescataría chb06** — no es el cuello de botella.
Los que ayudarían a chb06 específicamente son datos: más sujetos con crisis
cortas/sutiles (para aprender esa morfología), features espectrales (la crisis puede
estar en la frecuencia, no en la amplitud) y calibración por paciente.

## Hallazgo secundario: **mala calibración global**

chb09 "funciona" pero su interictal tiene score mediano 0.969: a umbral 0.5 hay
**10.916 falsos positivos** (spec 0.18). Solo sirve al umbral ~1.0. El `pos_weight=15`
+ el shift de dominio del normalizador LOSO dejan al modelo sistemáticamente
sobre-confiado. **El umbral óptimo es fuertemente por-paciente** — un umbral fijo
global es inservible. Esto vale para todos los folds, no solo chb09.
