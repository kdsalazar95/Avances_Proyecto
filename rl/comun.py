"""Utilidades compartidas por el agente y el evaluador del ciclo RL.

Solo funciones deterministas: rutas del repositorio, lectura/escritura de
JSON, hashes de archivos y acceso a los módulos del simulador existente
(``cgra/src``), que se importan tal cual, sin modificarlos.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent
RAIZ_CGRA = RAIZ_REPO / "cgra"
SRC_CGRA = RAIZ_CGRA / "src"
RUN_CGRA = SRC_CGRA / "run_cgra.py"
FUENTE_OPERACIONES = RAIZ_CGRA / "operaciones" / "operaciones.c"
ESQUEMA_CGRA = SRC_CGRA / "cgra.json"
INSTRUCCIONES_AGENTE = Path(__file__).resolve().parent / "agente" / "instrucciones_agente.md"
SKILLS_01_09 = RAIZ_CGRA / "agent_skills"

# Acción del agente: cgra.json (+ candidata.json con su justificación).
# Programas que generan los Skills 01–09 dentro del evaluador: pe_instructions/ y manifest.json.
ARCHIVO_CONFIG = "cgra.json"
ARCHIVO_MANIFEST = "manifest.json"
ARCHIVO_METADATOS = "candidata.json"
CARPETA_PROGRAMAS = "pe_instructions"


def usar_simulador():
    """Hace importables los módulos de ``cgra/src`` (FormatoCGRA, Cargador, ...)."""
    ruta = str(SRC_CGRA)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def escribir_json(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ruta


def agregar_jsonl(ruta, datos):
    """Agrega una línea a un archivo JSONL (solo agregar, nunca reescribir)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(datos, ensure_ascii=False) + "\n")


def leer_jsonl(ruta):
    ruta = Path(ruta)
    if not ruta.is_file():
        return []
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def sha256_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def hashes_paquete(directorio):
    """Hashes de los artefactos de una candidata: cgra.json, manifest y cada PE*.txt.

    Las claves son rutas relativas (``cgra.json``, ``pe_instructions/PE00.txt``).
    """
    directorio = Path(directorio)
    hashes = {}
    for nombre in (ARCHIVO_CONFIG, ARCHIVO_MANIFEST):
        if (directorio / nombre).is_file():
            hashes[nombre] = sha256_archivo(directorio / nombre)
    programas = directorio / CARPETA_PROGRAMAS
    if programas.is_dir():
        for ruta in sorted(programas.glob("*.txt")):
            hashes[f"{CARPETA_PROGRAMAS}/{ruta.name}"] = sha256_archivo(ruta)
    return hashes


def hash_accion(hashes):
    """Hash de la acción del agente: su cgra.json. Identifica configuraciones repetidas."""
    return hashes.get(ARCHIVO_CONFIG)


def hash_conjunto(hashes):
    """Un solo hash para un conjunto de archivos (p. ej. los PE*.txt generados)."""
    partes = [f"{ruta}:{valor}" for ruta, valor in sorted(hashes.items())]
    return hashlib.sha256("\n".join(partes).encode("utf-8")).hexdigest()


def ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relativa(ruta, base=RAIZ_REPO):
    """Ruta relativa a ``base`` si es posible (para reportes legibles)."""
    ruta = Path(ruta).resolve()
    try:
        return str(ruta.relative_to(Path(base).resolve()))
    except ValueError:
        return str(ruta)
