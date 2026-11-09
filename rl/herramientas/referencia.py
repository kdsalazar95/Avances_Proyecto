"""Escribe un ``cgra.json`` (una acción del agente) con filas ``M X M`` o ``M X``.

``X`` es el PE de cómputo de cada fila: E (escalar), V (vectorial) o C
(escalar + vectorial). Sirve, por ejemplo, para la referencia 3x3
``M E M / M V M / M E M`` de la corrida (``uv run python -m rl referencia``).
"""

import argparse
import json
from pathlib import Path

from ..comun import ARCHIVO_CONFIG, ARCHIVO_METADATOS, ESQUEMA_CGRA


def _esquema(esquema=ESQUEMA_CGRA):
    return json.loads(Path(esquema).read_text(encoding="utf-8"))


def malla_filas(computo, con_este=True):
    """``["E", "V", "E"]`` -> ``["M E M", "M V M", "M E M"]`` (o ``"M E"`` sin columna este)."""
    return [f"M {x} M" if con_este else f"M {x}" for x in computo]


def escribir_cgra(salida, malla, esquema=ESQUEMA_CGRA, justificacion=None, id_candidata=None):
    """Escribe ``cgra.json`` (y ``candidata.json``) con la ``malla`` dada (lista de cadenas)."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    datos = _esquema(esquema)
    filas = [fila.split() for fila in malla]
    letras = sorted({letra for fila in filas for letra in fila})
    cgra = {
        "descripcion": f"Malla {len(filas)}x{len(filas[0])}: " + " / ".join(malla),
        "filas": len(filas),
        "columnas": len(filas[0]),
        "palabras_memoria": datos["palabras_memoria"],
        "ancho_vector": datos["ancho_vector"],
        "tipos_pe": {letra: datos["tipos_pe"][letra] for letra in letras},
        "malla": list(malla),
    }
    (salida / ARCHIVO_CONFIG).write_text(json.dumps(cgra, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadatos = {"id_candidata": id_candidata or salida.name, "justificacion": justificacion or cgra["descripcion"]}
    (salida / ARCHIVO_METADATOS).write_text(json.dumps(metadatos, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return salida / ARCHIVO_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Escribe un cgra.json de filas 'M X M' (o 'M X').")
    parser.add_argument("salida")
    parser.add_argument("--computo", default="E,V,E", help="tipo de cómputo de cada fila, p. ej. E,V,E")
    parser.add_argument("--sin-este", action="store_true")
    argumentos = parser.parse_args()
    computo = argumentos.computo.split(",")
    print(escribir_cgra(argumentos.salida, malla_filas(computo, not argumentos.sin_este)))


if __name__ == "__main__":
    main()
