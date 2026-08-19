# Comandos — referencia rápida

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../LEGAL.md).

> Todos los comandos importantes del proyecto en un solo lugar. Se corren desde la raíz
> del repo. Asumen el dataset CHB-MIT en `data/chb-mit/` (cambialo con `--data-root`).

## Instalación

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # instala el paquete `neuropilot` (editable)
```

## Usar el detector (inferencia)

```bash
# Web local: subís un .edf y ves detección + explicación en el navegador
uvicorn app.main:app --host 127.0.0.1 --port 8000     # abrí http://127.0.0.1:8000

# CLI: detectar sobre un archivo, con explicabilidad
python -m neuropilot.inference.detector registro.edf --explain

# CLI con un modelo adaptado a un paciente + umbral para crisis cortas
python -m neuropilot.inference.detector registro.edf \
    --model models/chbNN_adapted.pt --min-consecutive 1
```

## Adaptar el modelo a un paciente (few-shot)

```bash
# 1) adaptar con unas grabaciones del paciente (crisis ya etiquetadas en su summary)
python -m neuropilot.inference.adapt \
    --patient-dir data/chb-mit/chbNN --files chbNN_01.edf chbNN_03.edf \
    --out models/chbNN_adapted.pt

# 2) detectar en registros nuevos de ese paciente con el modelo adaptado
python -m neuropilot.inference.detector data/chb-mit/chbNN/chbNN_20.edf \
    --model models/chbNN_adapted.pt
```

Explicación de qué hace: [adaptacion.md](adaptacion.md).

## Entrenar

```bash
# Evaluación LOSO (entrena N modelos, uno por paciente dejado afuera). Reanudable.
python -m neuropilot.training.run_loso \
    --data-root data/chb-mit --out experiments/exp_005 \
    --patients chb01 chb02 chb03 chb04 chb05 chb06 chb07 chb08 chb09 --epochs 15

# Modelo de despliegue (uno solo, todos los pacientes) -> models/deployment_v1.pt
python -m neuropilot.training.train_deployment \
    --cache-dir experiments/exp_005/cache \
    --patients chb01 chb02 chb03 chb04 chb05 chb06 chb07 chb08 chb09 chb10 chb11 \
    --out models/deployment_v1.pt
```

> El cache de ventanas (`experiments/exp_005/cache/`, ~20 GB) se genera una vez y se reusa
> con `--cache-dir`. No se versiona (está en `.gitignore`).

## Experimentos y análisis

```bash
python experiments/exp_005/viability_eval.py        # cobertura por evento vs FA/hora
python experiments/exp_005/calibration_eval.py      # umbral por paciente (label-free)
python experiments/exp_005/error_analysis.py        # por qué chb06 falla
python experiments/exp_005/hybrid_eval.py           # híbrido CNN + baseline (rechazado)
python experiments/exp_005/second_stage_eval.py     # 2da etapa anti-FA (rechazada)
python experiments/exp_008_patient_adapt/run.py     # adaptación al paciente (chb12)
python experiments/exp_008_patient_adapt/chb06.py   # confirmación en chb06
```

## Tests

```bash
python -m pytest -q          # toda la suite (120 tests)
python -m pytest tests/test_events.py -q
```

## Descargar el dataset (CHB-MIT)

```bash
# ejemplo: bajar chb01 desde PhysioNet a data/chb-mit/
mkdir -p data/chb-mit && cd data/chb-mit
wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" \
    https://physionet.org/files/chbmit/1.0.0/chb01/
```
