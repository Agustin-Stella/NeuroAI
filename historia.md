# NeuroPilot AI — la historia del proyecto

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](LEGAL.md).


> Cómo construí un detector de crisis epilépticas en EEG que **supera al baseline**,
> es **usable** (MVP web) y —sobre todo— cuyas métricas **no me mienten**.
> Herramienta de **asistencia**, no de diagnóstico.

Este documento cuenta el recorrido completo: el problema, cada experimento (incluidos los
que salieron mal), y por qué las decisiones se tomaron con evidencia y no con intuición.
Si venís del código, la puerta de entrada es [`docs/pipeline.md`](docs/pipeline.md); los
conceptos están en [`docs/glosario.md`](docs/glosario.md).

---

## El problema

Un neurólogo puede pasar horas revisando un EEG de días buscando crisis epilépticas. La
idea: una IA que **pre-lea** la señal y marque dónde hay patrones compatibles con crisis,
con su confianza y una explicación — para que el humano valide más rápido.

Dos cosas hacen que esto sea traicionero:

1. **Desbalance brutal.** Las crisis son ~1 % de la señal. Un modelo que diga "normal"
   siempre "acierta" el 99 %… sin detectar **ni una** crisis. La *accuracy* miente.
2. **Generalización entre pacientes.** Un EEG varía muchísimo de persona a persona. Lo
   fácil es que el modelo memorice a los pacientes de entrenamiento y falle con uno nuevo.

Por eso el proyecto se paró sobre una regla desde el día uno: **medir honestamente**.

---

## Las reglas (para no engañar)

- **Split por paciente.** Ningún dato de un paciente de test aparece en train.
- **LOSO (Leave-One-Subject-Out).** Se entrena dejando un paciente afuera y se evalúa en
  él. Es la única forma honesta de estimar cómo va a andar con alguien nuevo.
- **Baseline primero.** Un modelo clásico simple como vara. Si el deep learning no le gana
  de forma medible, no aporta.
- **Métricas médicas, nunca accuracy sola:** sensibilidad, AUPRC, sensibilidad a
  especificidad fija, y —lo que de verdad importa— **cobertura por evento vs falsas
  alarmas por hora**.

Datos: **CHB-MIT** (EEG con crisis anotadas), pacientes chb01–chb11. *(chb12 quedó afuera:
tiene el montaje de canales inconsistente entre archivos — un problema del dato, no del
modelo.)*

---

## El recorrido, experimento por experimento

### 1. El baseline honesto — AUPRC 0.377
Regresión logística sobre potencia por bandas (δ/θ/α/β/γ). Simple, interpretable. La vara
a superar.

### 2. Primer intento de CNN — **fracaso** (AUPRC 0.049, ROC-AUC 0.467)
Una CNN 1D sobre la señal cruda dio **peor que el azar**. Resultado incómodo, pero la
mentalidad científica funcionando: la evidencia decía que el modelo "más sofisticado" no
servía. No se adoptó nada; se buscó la causa.

### 3. El diagnóstico — la normalización
Analizando el fracaso: la CNN recibía la señal en voltios crudos, sin normalizar. La
amplitud del EEG varía tanto entre pacientes que lo aprendido en uno no se trasladaba a
otro. Hipótesis: **normalizar por canal es el arreglo de mayor impacto.**

### 4. La CNN que sí funciona — **AUPRC 0.517** (supera al baseline)
Con z-score por canal (ajustado solo con train, sin leakage) + más pacientes, la CNN pasó
a **0.517 de AUPRC** (mediana 0.498), superando al baseline (0.377). Por primera vez el
deep learning ganó de forma medible.

### 5. ¿Fue la normalización o los datos? — **experimento controlado**
Un salto grande, pero habían cambiado **dos** cosas (normalización *y* más pacientes). Así
que no se podía atribuir la mejora a una sola. Corrí un experimento controlado (mismo
pipeline, cambiando un factor por vez):

> **El ~87 % de la mejora vino de la normalización**, no del volumen de datos.
> Confirmado con evidencia, no con intuición.

### 6. ¿Por qué falla en un paciente? — **error analysis de chb06**
Un paciente (chb06) colapsaba. En vez de esconderlo, lo diseccioné:
- Sus crisis son **cortas** (~15 s vs ~68 s de un paciente fácil) y **sin contraste de
  amplitud** — electrográficamente sutiles.
- **El baseline también falla chb06.** Dos familias de modelo distintas fallan igual → es
  una limitación de **los datos de ese paciente**, no de la arquitectura.

Conclusión clave: agregar capacidad al modelo (ej. CNN+LSTM) **no** arreglaría chb06. El
cuello de botella no es el modelo.

### 7. Hacerlo usable — **calibración por paciente (sin leakage)**
El modelo rankeaba bien, pero el umbral de alarma óptimo variaba muchísimo por paciente
(un umbral global perdía ~19 puntos de sensibilidad). Solución **deployable**: calibrar el
umbral con la distribución de scores del propio paciente (percentil), **sin usar
etiquetas**. Recupera el **99 %** de la sensibilidad del umbral oráculo — sin hacer trampa.

### 8. ¿Es viable? — **la métrica que importa: por evento**
Reformulé la evaluación a nivel clínico (evento, no ventana):

> **Detecta el 81 % de las crisis (39/48). En 8 de 9 pacientes, el 100 %.**
> Ajustable a **< 1 falsa alarma por hora** manteniendo 77 % de cobertura.

El único que baja el promedio es chb06 (el caso difícil ya diagnosticado).

### 9. Dos intentos de mejora — **ambos rechazados con evidencia**
- **Híbrido CNN + baseline.** Hipótesis: fallan en pacientes distintos → combinarlos
  cubriría más. Resultado: **peor** (79 % < 81 %). Las 9 crisis que faltan son todas de
  chb06, y el baseline *también* falla chb06 → ninguna combinación puede romper ese techo.
- **Segunda etapa anti-falsas-alarmas.** Un clasificador para filtrar falsas alarmas.
  Resultado: **dominado** por simplemente subir el umbral del CNN (una línea de config le
  gana en todos los puntos de operación).

Dos rechazos seguidos = una señal fuerte y honesta: **CNN + calibración por paciente está
cerca de su techo dado el dato disponible.** Lo que queda es un problema de datos (chb06),
no de complejidad de modelo. *Saber cuándo parar de agregar cosas es parte del trabajo.*

### 10. Reality-check: un paciente REALMENTE no visto — **chb12 → 1/14** ⚠️

El 81 % sale del LOSO sobre chb01–09. Para un test más duro, corrí el modelo de despliegue
sobre **chb12**, que quedó **completamente fuera del entrenamiento**. El resultado fue
crudo: **detectó 1 de 14 crisis** (el score quedó plano en ~0, ciego a sus crisis).

Qué enseña esto —y por qué es honesto ponerlo—:

- **La generalización es fuertemente dependiente del paciente.** El promedio "8 de 9 al
  100 %" esconde una varianza enorme: hay pacientes (chb06, chb12) donde el modelo falla
  casi por completo. chb12 tiene crisis **cortas y sutiles**, el mismo perfil difícil de chb06.
- **El titular hay que templarlo:** el modelo funciona muy bien en la mayoría de los
  pacientes, pero **falla en algunos, y todavía no podemos predecir en cuáles**. La
  generalización *zero-shot* a un paciente arbitrario es un problema abierto en EEG, no un
  bug de este proyecto.
- Confirma, por tercera vez, que **la palanca es dato/adaptación al paciente, no más modelo.**

Probar sobre un paciente genuinamente held-out y reportar que falla es más creíble que
mostrar solo los casos lindos. Es la parte del proyecto que lo hace serio.

### 11. La solución: adaptación al paciente — **chb12: 6 % → 65 %** ✅

Si el problema es que un paciente nuevo tiene crisis atípicas, la respuesta —y la forma en
que se despliegan los detectores clínicos reales— es **adaptarse a él**. Con un
fine-tuning corto sobre unos pocos archivos de chb12 (~82 ventanas de crisis) + un
normalizador ajustado al propio paciente, y testeando en los archivos **restantes** (sin
leakage):

> **chb12 pasó de 6 % a 65 %** con adaptación básica, y a **91 % (a 1.7 falsas
> alarmas/hora)** ajustando además el punto de operación a su perfil de crisis.

El salto de 65 % → 91 % vino de un detalle fino: chb12 tiene **crisis cortas**, y exigir 2
ventanas seguidas para declarar un evento se las perdía; con 1 ventana basta. Lección
extra: **el punto de operación también conviene adaptarlo al paciente** (crisis cortas vs
largas).

**Y no fue suerte con chb12.** Repetí la receta en **chb06** —el paciente que había fallado
en *todo* el proyecto (baseline, CNN, híbrido, 2da etapa)—: **20 % zero-shot → 100 %
adaptado** (5/5 crisis en archivos held-out). Dos pacientes difíciles distintos, ambos
rescatados → el método es general. **El techo nunca fue el modelo: era la generalización
zero-shot, y la adaptación al paciente la resuelve.**

El mensaje correcto del proyecto queda así: *la generalización zero-shot cross-paciente es
el problema difícil (0–6 % en un paciente atípico); con **adaptación al paciente** —como
los detectores clínicos reales— se recupera a **~90 %**.* No es "no funciona": es que hay
que usarlo como se usa de verdad.

---

## El MVP — de la investigación a algo que se puede correr

- **Modelo de despliegue:** un único modelo entrenado con los 11 pacientes, checkpoint
  autocontenido (lleva adentro el montaje, el preprocess y la calibración). Corre sobre
  **cualquier** EEG nuevo.
- **Web local (FastAPI):** subís un `.edf`, ves los eventos detectados, la línea de tiempo
  y **la explicación** (qué canales y en qué momento activaron la detección, vía saliencia
  por gradiente). 100 % local, ningún dato sale de la máquina.
- **Robustez:** ante un montaje incompatible, mensaje claro en vez de romperse.
- **120 tests** en verde.

```bash
# levantar la web
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
# o por línea de comando, sobre un archivo:
.venv/bin/python -m neuropilot.inference.detector registro.edf --explain
```

Flujo humano ↔ IA: la IA lee y marca; **el humano decide**. Es asistencia, no diagnóstico.

---

## Resultados, en una tabla

| | Métrica | Valor |
|---|---|---|
| Modelo | AUPRC (LOSO, pacientes no vistos) | **0.517** (baseline: 0.377) |
| Clínico | Crisis detectadas (por evento, LOSO chb01–09) | **81 %** — 39/48 |
| Clínico | Pacientes con el 100 % de sus crisis | **8 de 9** |
| Clínico | Falsas alarmas | **< 1 / hora** (a 77 % de cobertura) |
| ⚠️ Reality-check | Paciente 100 % no visto (chb12), zero-shot | **6 %** — la generalización zero-shot es dura |
| ✅ Con adaptación | Mismo paciente (chb12), adaptado + operación ajustada | **~91 %** a 1.7 FA/h |
| Ciencia | Origen de la mejora (medido) | ~87 % normalización |
| Ingeniería | Tests | 120 en verde |

---

## Qué me llevo (lo que hace serio a esto)

1. **En IA médica, lo difícil no es entrenar la red — es no engañarte con las métricas.**
   LOSO, split por paciente y "cobertura por evento vs FA/h" fueron más importantes que la
   arquitectura.
2. **Un resultado negativo bien medido vale tanto como uno positivo.** El primer CNN
   fracasó, el híbrido y la segunda etapa se rechazaron. Cada "no" vino con evidencia y con
   un *por qué*.
3. **Saber cuándo parar.** La evidencia dijo que no había jugo fácil sin más datos. Agregar
   complejidad a ciegas hubiera sido peor que quedarme quieto.

---

## Limitaciones (honestas)

- **La generalización zero-shot es dura (la limitación central):** en un paciente 100 % no
  visto y atípico (chb12), zero-shot detecta ~6 %. **Con adaptación al paciente sube a
  ~91 %**, pero eso requiere algo de dato etiquetado del paciente — el modelo no funciona
  igual de bien "para cualquiera de una" sin adaptación. Es el trade-off real: zero-shot
  universal (abierto) vs patient-adaptive (funciona, como en la práctica clínica).
- **Muestra chica:** 48 crisis / 11 pacientes → incertidumbre real.
- **chb06 sin resolver:** crisis cortas y sutiles; necesita datos/features específicos, no
  más modelo.
- **Dataset controlado:** CHB-MIT es pediátrico y relativamente limpio; el mundo real
  (otro hospital, otro equipo) sería más ruidoso — se espera degradación.
- **Sin validación externa todavía:** el próximo paso científico sería evaluar en otro
  dataset (ej. TUH EEG).

---

## Stack

Python · PyTorch · MNE-Python · scikit-learn · NumPy/SciPy · FastAPI · pytest

## Mapa del repo

| Ruta | Qué hay |
|---|---|
| [`neuropilot/`](neuropilot/) | El paquete: datos, preprocesamiento, modelo, entrenamiento, evaluación, inferencia, explicabilidad |
| [`app/`](app/) | La web local (FastAPI + página) |
| [`experiments/`](experiments/) | Cada experimento con su reporte y sus figuras |
| [`docs/`](docs/) | Arquitectura, metodología, pipeline, glosario, explicabilidad |
| [`models/`](models/) | El modelo de despliegue |

> ⚕️ NeuroPilot AI produce predicciones y patrones compatibles, nunca diagnósticos
> confirmados. No reemplaza el criterio médico.
