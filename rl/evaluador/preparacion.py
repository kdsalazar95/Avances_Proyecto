"""E01 · Recepción de la configuración del agente.

Crea una carpeta aislada por evaluación y copia en ella, sin modificarlo, el
``cgra.json`` de la candidata y la memoria inicial de la instancia (una copia
intacta y otra de trabajo para ``--write-back``). Los programas los generan
después los Skills 01–09 (``compilador.py``) dentro de esta carpeta.
"""

import shutil
import time
from pathlib import Path

from ..comun import ARCHIVO_CONFIG, sha256_archivo


def recibir(dir_candidata, instancia, carpeta_eval, id_evaluacion):
    """Devuelve el resultado de E01. Lo que falta de la instancia es de infraestructura."""
    inicio = time.monotonic()
    dir_candidata, carpeta_eval = Path(dir_candidata), Path(carpeta_eval)
    errores = []
    faltan_instancia = [clave for clave in ("memoria_inicial", "esperado")
                        if not Path(instancia.get(clave, "")).is_file()]
    if faltan_instancia:
        errores.append({"skill": "E01", "regla": "faltantes", "detalle": f"faltan: {', '.join(faltan_instancia)}"})
    if not (dir_candidata / ARCHIVO_CONFIG).is_file():
        errores.append({"skill": "E01", "regla": "faltantes", "detalle": "la candidata no tiene cgra.json"})
    if carpeta_eval.exists():
        errores.append({"skill": "E01", "regla": "carpeta_existente",
                        "detalle": f"la carpeta de evaluación ya existe: {carpeta_eval}"})
    if errores:
        return {"listo": False, "errores": errores, "inicio_monotonic": inicio, "id_evaluacion": id_evaluacion}

    carpeta_eval.mkdir(parents=True)
    shutil.copy2(dir_candidata / ARCHIVO_CONFIG, carpeta_eval / ARCHIVO_CONFIG)
    shutil.copy2(instancia["memoria_inicial"], carpeta_eval / "memoria_inicial.bin")
    shutil.copy2(instancia["memoria_inicial"], carpeta_eval / "memoria.bin")
    if sha256_archivo(carpeta_eval / ARCHIVO_CONFIG) != sha256_archivo(dir_candidata / ARCHIVO_CONFIG):
        errores.append({"skill": "E01", "regla": "copia_alterada", "detalle": "la copia de cgra.json no coincide"})
    if sha256_archivo(carpeta_eval / "memoria_inicial.bin") != instancia["hash_memoria_inicial"]:
        errores.append({"skill": "E01", "regla": "memoria_alterada",
                        "detalle": "la memoria inicial no coincide con la de la instancia"})
    return {"listo": not errores, "errores": errores, "carpeta": str(carpeta_eval),
            "inicio_monotonic": inicio, "id_evaluacion": id_evaluacion}
