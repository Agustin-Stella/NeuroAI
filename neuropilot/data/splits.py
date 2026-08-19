"""Splits por paciente — el invariante no negociable del proyecto.

Regla dura: **ninguna ventana/archivo de un paciente puede aparecer en más de un
split**. El split se hace por *sujeto*, no por archivo ni por ventana. Si un
paciente está en test, TODOS sus datos están en test. Esto evita el error #1 que
hunde proyectos de EEG: un modelo que memoriza al paciente en vez de la patología
y muestra métricas infladas que se evaporan con datos nuevos.

Este módulo:
  - construye un split train/val/test por paciente de forma **determinista** (seed),
  - lo **congela** en un JSON versionado (ej. ``splits/v1.json``) para que todos los
    experimentos usen exactamente la misma partición,
  - genera folds **LOSO** (Leave-One-Subject-Out) para evaluación honesta con pocos
    sujetos.

No hace preprocessing ni lee señal: opera sobre identificadores de paciente.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "neuropilot.patient_split/v1"

_RE_PATIENT_ID = re.compile(r"^(chb\d+)", re.IGNORECASE)


def patient_id_from_filename(file_name: str) -> str:
    """Deriva el id de paciente de un archivo CHB-MIT.

    ``"chb01_03.edf" -> "chb01"``. Sirve para agrupar archivos por sujeto antes
    de asignarlos a un split.
    """
    match = _RE_PATIENT_ID.match(Path(file_name).name)
    if not match:
        raise ValueError(f"No pude extraer patient_id de: {file_name!r}")
    return match.group(1).lower()


# --------------------------------------------------------------------------- #
# Split train/val/test por paciente
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatientSplit:
    """Partición de pacientes en train / val / test.

    Cada lista contiene ids de paciente. Las tres son disjuntas (se valida al
    construir): esa disjunción ES el invariante que protege contra leakage.
    """

    train: list[str]
    val: list[str]
    test: list[str]

    def __post_init__(self) -> None:
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        # No duplicados dentro de una misma lista.
        for name, s in sets.items():
            if len(s) != len(getattr(self, name)):
                raise ValueError(f"Pacientes duplicados en '{name}'")
        # Disjunción entre listas (el invariante).
        overlap_tv = sets["train"] & sets["val"]
        overlap_tt = sets["train"] & sets["test"]
        overlap_vt = sets["val"] & sets["test"]
        leaked = overlap_tv | overlap_tt | overlap_vt
        if leaked:
            raise ValueError(
                f"LEAKAGE por paciente: {sorted(leaked)} aparece en más de un split"
            )

    @property
    def all_patients(self) -> list[str]:
        return [*self.train, *self.val, *self.test]

    def split_of(self, patient_id: str) -> str:
        """Devuelve 'train' | 'val' | 'test' para un paciente, o error si no está."""
        pid = patient_id.lower()
        if pid in set(self.train):
            return "train"
        if pid in set(self.val):
            return "val"
        if pid in set(self.test):
            return "test"
        raise KeyError(f"Paciente {patient_id!r} no está en el split")

    def to_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, data: dict) -> "PatientSplit":
        return cls(train=list(data["train"]), val=list(data["val"]), test=list(data["test"]))


def make_patient_split(
    patients: list[str],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> PatientSplit:
    """Construye un split por paciente determinista.

    El resultado depende SOLO de ``patients`` (como conjunto) y ``seed``: se ordena
    primero y se baraja con la semilla, así el mismo seed da el mismo split sin
    importar el orden de entrada.

    Parameters
    ----------
    patients  : ids de paciente (se deduplican y ordenan).
    val_frac  : fracción de pacientes para validación.
    test_frac : fracción de pacientes para test.
    seed      : semilla de reproducibilidad.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac y test_frac deben ser >= 0 y sumar < 1")

    unique = sorted({p.lower() for p in patients})
    n = len(unique)
    if n < 3:
        raise ValueError(f"Se necesitan al menos 3 pacientes para un split; hay {n}")

    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(test_frac * n)) if test_frac > 0 else 0
    n_val = max(1, round(val_frac * n)) if val_frac > 0 else 0
    if n - n_test - n_val < 1:
        raise ValueError(
            f"Fracciones demasiado grandes para {n} pacientes "
            f"(test={n_test}, val={n_val}, train quedaría vacío)"
        )

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]

    # Orden estable dentro de cada bucket para diffs limpios en el JSON.
    return PatientSplit(train=sorted(train), val=sorted(val), test=sorted(test))


# --------------------------------------------------------------------------- #
# Congelar / cargar el split (JSON versionado)
# --------------------------------------------------------------------------- #
def save_split(
    split: PatientSplit,
    path: str | Path,
    *,
    seed: int | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    extra: dict | None = None,
) -> Path:
    """Congela un split en JSON versionado (ej. ``splits/v1.json``).

    Guarda también metadata de reproducibilidad (schema, seed, fracciones,
    timestamp, cantidad de pacientes). ``extra`` permite anexar campos propios
    (por ejemplo, el hash de git del código que lo generó).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_patients": len(split.all_patients),
        "counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        **split.to_dict(),
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: str | Path) -> PatientSplit:
    """Carga un split congelado desde JSON. Valida el schema."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"Schema inesperado: {schema!r} (esperaba {SCHEMA!r})")
    return PatientSplit.from_dict(data)


# --------------------------------------------------------------------------- #
# LOSO — Leave-One-Subject-Out
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LosoFold:
    """Un fold LOSO: un paciente de test, el resto de train."""

    test_patient: str
    train_patients: list[str]


def loso_folds(patients: list[str]) -> list[LosoFold]:
    """Genera folds Leave-One-Subject-Out.

    Con pocos sujetos (CHB-MIT ~24), un único split tiene varianza enorme; LOSO
    entrena N modelos, cada uno evaluado sobre un paciente distinto no visto, y se
    reporta media ± desvío entre sujetos.
    """
    unique = sorted({p.lower() for p in patients})
    if len(unique) < 2:
        raise ValueError("LOSO necesita al menos 2 pacientes")
    folds: list[LosoFold] = []
    for test_patient in unique:
        train = [p for p in unique if p != test_patient]
        folds.append(LosoFold(test_patient=test_patient, train_patients=train))
    return folds
