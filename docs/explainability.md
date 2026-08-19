# Explicabilidad y visualización para el médico

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).


> Decisión de producto sobre **qué ve el médico**, no solo qué calcula el modelo.
> Complementa el principio de explicabilidad del proyecto ("no crear cajas negras") y la
> sección "la parte humana" de `docs/research.md`.

---

## El punto

El reporte de texto **no alcanza**. Un médico no confía en una caja negra que dice
"89% de probabilidad de crisis". La herramienta tiene que **mostrarle algo que él
pueda verificar** con su propio criterio, sobre una representación que ya sabe leer.

Pero mostrar lo correcto importa: hay dos visualizaciones distintas para dos
públicos distintos.

---

## Dos visualizaciones, dos públicos

### 1. Heatmap de potencia por banda — herramienta INTERNA
"Lo que ve la IA" (ej. `experiments/exp_001/lo_que_ve_la_ia.png`): potencia por
banda (δ/θ/α/β/γ) por ventana. **Es para el equipo de desarrollo**, para entender
y depurar qué features usa el modelo. Un neurólogo no lee EEG en bandas promediadas
— este gráfico NO va en la interfaz clínica.

### 2. Señal cruda + explicación — para el MÉDICO
El médico ya es experto leyendo EEG crudo. La IA lo **asiste**, no lo reemplaza.

---

## Las 3 capas que necesita el médico (de menos a más útil)

| Capa | Qué es | Por qué le sirve |
|---|---|---|
| **Reporte** | Texto: "3 eventos, principal 13:42–13:43, confianza 89%" | Resumen rápido para el informe |
| **Señal + marcas** | EEG crudo con las zonas detectadas resaltadas (ver `crisis_chb01_03.png`) | Verifica de un vistazo, sobre algo que ya sabe leer |
| **El "por qué"** | Qué canales y en qué momento pesaron en la decisión (saliency / atención / SHAP) | Construye confianza: "hay crisis **por esto**", no solo "hay crisis" |

Las capas 2 y 3 son la clave del valor. Con ellas, el médico confirma una detección
en segundos en vez de leer 40 minutos de EEG. Sin ellas, no adopta la herramienta.

---

## Decisión

En la interfaz clínica (Etapa 5) se muestran las **tres capas juntas**:
1. Señal cruda con las detecciones resaltadas,
2. overlay del "por qué" (explicabilidad),
3. reporte de texto como resumen.

El heatmap de bandas queda como **herramienta interna** de desarrollo.

---

## Dónde encaja en el roadmap

- **Capa 2 (señal + marcas):** sencilla, ya prototipada (`crisis_chb01_03.png`).
- **Capa 3 (el "por qué"):** **Etapa 4 — explicabilidad** (SHAP, saliency/Grad-CAM,
  mapas de atención). Es el principio "explicabilidad primero".
- **Interfaz clínica:** **Etapa 5 — producto**.

---

## Pendiente crítico (la parte humana)

**Esto lo valida un médico real, no el equipo.** Antes de invertir en la Etapa 4/5,
preguntarle a un neurólogo / estudiante de medicina / investigador:

- ¿Le alcanza con la señal resaltada, o necesita sí o sí el "por qué"?
- ¿Qué forma del "por qué" le resulta creíble (canales, tiempo, frecuencia)?
- ¿El reporte de texto le ahorra tiempo real, o el cuello de botella es otro
  (ej. escribir el informe, comparar con estudios previos)?

Ese feedback puede reordenar las prioridades del producto.
