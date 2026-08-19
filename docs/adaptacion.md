# Adaptación al paciente — qué es y cómo se hace

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).

> Por qué el modelo general falla en algunos pacientes y cómo **adaptarlo** los rescata
> (chb06: 20 %→100 %, chb12: 6 %→91 %). Concepto + mecánica concreta.
> Complementa: [`pipeline.md`](pipeline.md), [`glosario.md`](glosario.md) y el
> experimento [`experiments/exp_008_patient_adapt/`](../experiments/exp_008_patient_adapt/).

## Qué es (en una frase)

**Adaptar = agarrar el modelo general y ajustarlo a UN paciente**, usando un poco de dato
de ese paciente. Deja de ser "un modelo para todos" y pasa a ser "el modelo de Juan".

Analogía: como un audífono o el reconocimiento de voz del celular — viene con una config
general que anda "más o menos", pero si lo **calibrás a vos** funciona muchísimo mejor.

## Por qué hace falta

El modelo general aprendió de muchos pacientes → conoce la crisis **típica**. Pacientes
como chb06/chb12 tienen crisis **atípicas** (cortas, sutiles) que no se parecen a lo
típico → para el modelo general son casi invisibles (6–20 %). No es que el modelo sea malo;
nunca vio ese "estilo" de crisis.

## Cómo se hace, concretamente

Necesitás **unas pocas crisis ya etiquetadas de ese paciente** (en la práctica siempre las
tenés: son sus primeros registros, que un neurólogo ya marcó). Con eso, tres mecanismos:

### 1. Fine-tuning (ajustar los "pesos")

La clave: **no se entrena de cero.** Se arranca de los pesos del modelo ya entrenado y se
le dan **unas pocas vueltas más, solo con el dato del paciente**, con pasos chiquitos:

- **Arrancar de los pesos entrenados** (no aleatorios) → el modelo ya sabe lo general.
- **Learning rate bajo** (`5e-4`) → pasos chicos: especializa sin "olvidar" lo aprendido.
- **Pocas epochs** (`~10`) → un empujón, no un reentrenamiento.
- **Solo el dato del paciente** → el modelo corre sus perillas para reconocer *esta* forma
  de crisis.

### 2. Normalizador propio del paciente

Se recalcula la escala de la señal (z-score por canal) con **el EEG de ese paciente**, no
el promedio de todos. Así el modelo ve la señal en "las unidades correctas" para esa persona.

### 3. Punto de operación a su medida

Se ajusta el umbral/agrupación a su perfil: crisis **cortas** → basta 1 ventana para marcar
(`min_consecutive=1`); crisis **largas** → se exige más para reducir falsas alarmas. (Esto
llevó chb12 de 65 % a 91 %.)

## En el código — [`neuropilot/inference/adapt.py`](../neuropilot/inference/adapt.py)

`adapt_to_patient(...)` hace, en orden:

| Paso | Función | Qué hace |
|---|---|---|
| 1 | `_records` | Lee el `summary` del paciente → tiempos de sus crisis en los archivos elegidos |
| 2 | `_materialize` | Cada EDF → 23 canales → filtra → ventanas de 4 s → etiqueta (crisis/normal) |
| 3 | undersampling | Todas las ventanas de crisis + negativos 15:1 (balance) |
| 4 | `fit_normalizer` | Calcula el z-score **del propio paciente** |
| 5 | `_build_model(base)` | Carga los **pesos del modelo base** (no aleatorios) |
| 6 | `train_model(epochs=10, lr=5e-4)` | **Fine-tuning:** pocas epochs, LR bajo, solo dato del paciente |
| 7 | `torch.save` | Checkpoint **autocontenido** (pesos + normalizador del paciente + `min_consecutive=1`) |

## Cómo se usa

```bash
# 1) adaptar el modelo a un paciente con unas crisis suyas ya etiquetadas
.venv/bin/python -m neuropilot.inference.adapt \
    --patient-dir /ruta/chbNN --files chbNN_01.edf chbNN_03.edf \
    --out models/chbNN_adapted.pt

# 2) detectar en registros nuevos de ese paciente
.venv/bin/python -m neuropilot.inference.detector nuevo.edf --model models/chbNN_adapted.pt
```

## El "pero" honesto

La adaptación **necesita algo de dato etiquetado del paciente** — no se puede adaptar a
alguien de quien no hay nada. Pero eso es exactamente cómo funciona en la clínica: cuando
llega un paciente, ya hay registros previos con crisis marcadas. Por eso los detectores
reales son **patient-specific**, y por eso este enfoque es el realista.

## Resultados (evidencia)

| Paciente difícil | Zero-shot | Adaptado |
|---|---|---|
| chb12 | 6 % | 91 % |
| chb06 | 20 % | **100 %** |

Dos pacientes difíciles distintos, ambos rescatados → **el techo no era el modelo, era la
generalización zero-shot; la adaptación al paciente la resuelve.**
