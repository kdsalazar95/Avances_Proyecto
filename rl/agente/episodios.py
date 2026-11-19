"""Episodios, exploración (epsilon) y caché de evaluaciones del Sistema 1, como en el Sistema 2.

**Episodio.** Empieza en la referencia 32x32. En cada paso el LLM (o, con probabilidad epsilon, un paso al azar)
propone una configuración; se evalúa y pasa a ser la mejor del episodio solo si baja su costo. El episodio
termina con ``sin_mejora_episodio`` propuestas seguidas sin mejorar, cuando el LLM responde PARAR o al llegar a
``max_pasos_episodio``. La corrida repite episodios hasta ``max_episodios`` o hasta ``paciencia`` episodios
seguidos sin bajar el mejor costo de la corrida (la regla del Sistema 2).

**Exploración (opción A).** El paso al azar es un cambio pequeño de la mejor configuración del episodio, con los
mismos movimientos del Sistema 2: quitar un bloque de filas o de columnas del interior (16, 8, 4, 2 o 1, del
inicio o del final) o cambiar el tipo de un PE del interior (E <-> V). Primero se elige al azar el tipo de
movimiento (quitar un bloque o cambiar un PE) y después uno de esos movimientos, también al azar.

**Caché.** Una configuración ya evaluada no se vuelve a evaluar (como en el Sistema 2). La clave es el hardware
(tamaño, malla, tipos de PE y sus instrucciones, ancho vectorial y memoria), no el texto del cgra.json. Si ya salió
en esta corrida, su evaluación cuesta 0 s (``repetida``); si viene de otra corrida (la caché compartida del
barrido), cuenta lo que tardó cuando se evaluó.
"""

import json
from pathlib import Path

from .. import estados
from ..comun import ARCHIVO_CONFIG, escribir_json, leer_json

BLOQUES = (16, 8, 4, 2, 1)
MEMORIA_OPS = {"LD", "ST", "VLD", "VST"}  # las mismas de entorno/validacion_accion.py


# -- exploración: un paso al azar (opción A) ------------------------------------------------------------
def tipos_de(restricciones):
    """``(memoria, computo)``: las letras de los tipos que acceden a memoria y las de solo cómputo."""
    por_tipo = restricciones["instrucciones_por_tipo"]
    memoria = sorted(t for t, ins in por_tipo.items() if set(ins) & MEMORIA_OPS)
    computo = sorted(t for t, ins in por_tipo.items() if not set(ins) & MEMORIA_OPS)
    return memoria, computo


def movimientos(malla, restricciones):
    """Los movimientos posibles desde ``malla`` (lista de filas): ``{"bloque": [...], "tipo": [...]}``.

    ``bloque``: ``(nombre, nueva_malla)`` por cada bloque de filas o columnas del interior que se puede quitar
    (16, 8, 4, 2 o 1, del inicio o del final) sin bajar de los mínimos. Dos movimientos que dan la misma malla
    cuentan una vez (se queda el del final, como en el Sistema 2). ``tipo``: ``(nombre, nueva_malla)`` por
    cada PE del interior que puede cambiar a otro tipo de cómputo (E <-> V)."""
    filas = [f.split() for f in malla]
    n_filas, n_cols = len(filas), len(filas[0])
    min_filas = max(1, restricciones["filas"]["minimo"])
    min_cols = max(3, restricciones["columnas"]["minimo"])
    _, computo = tipos_de(restricciones)
    bloque, vistas = [], set()
    for b in BLOQUES:
        if n_filas - b >= min_filas:
            for lado, nuevas in (("FILAS FIN", filas[:-b]), ("FILAS INICIO", filas[b:])):
                m = tuple(" ".join(f) for f in nuevas)
                if m not in vistas:
                    vistas.add(m)
                    bloque.append((f"{lado} {b}", list(m)))
        if n_cols - b >= min_cols:  # columnas del interior: la 0 y la última (memoria) se quedan
            for lado, corte in (("COLUMNAS FIN", lambda f: f[:-1 - b] + [f[-1]]),
                                ("COLUMNAS INICIO", lambda f: [f[0]] + f[1 + b:])):
                m = tuple(" ".join(corte(f)) for f in filas)
                if m not in vistas:
                    vistas.add(m)
                    bloque.append((f"{lado} {b}", list(m)))
    tipo = []
    for i, fila in enumerate(filas):
        for j in range(1, n_cols - 1):
            for otro in computo:
                if otro != fila[j] and fila[j] in computo:
                    nueva = [list(f) for f in filas]
                    nueva[i][j] = otro
                    tipo.append((f"CAMBIAR {i} {j} {fila[j]}->{otro}", [" ".join(f) for f in nueva]))
    return {"bloque": bloque, "tipo": tipo}


def paso_al_azar(azar, malla, restricciones):
    """``(nombre, nueva_malla)``: un movimiento al azar desde ``malla`` (ver el docstring del módulo)."""
    opciones = movimientos(malla, restricciones)
    clases = [c for c in ("bloque", "tipo") if opciones[c]]
    if not clases:
        raise ValueError("no hay ningún movimiento posible desde esta malla")
    return azar.choice(opciones[azar.choice(clases)])


# -- caché de evaluaciones -----------------------------------------------------------------------------
def clave_hardware(directorio):
    """El hardware de un cgra.json, como texto canónico (sin la descripción ni la justificación)."""
    datos = leer_json(Path(directorio) / ARCHIVO_CONFIG)
    tipos = {letra: sorted(t.get("instrucciones", [])) for letra, t in (datos.get("tipos_pe") or {}).items()}
    return json.dumps({"filas": datos.get("filas"), "columnas": datos.get("columnas"), "malla": datos.get("malla"),
                       "tipos_pe": tipos, "ancho_vector": datos.get("ancho_vector"),
                       "palabras_memoria": datos.get("palabras_memoria")}, sort_keys=True)


class CacheEvaluaciones:
    """Caché compartida entre corridas (la del barrido): un reporte por configuración en ``carpeta/<hash>.json``."""

    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)

    def _ruta(self, clave):
        import hashlib
        return self.carpeta / f"{hashlib.sha256(clave.encode()).hexdigest()[:32]}.json"

    def buscar(self, clave):
        ruta = self._ruta(clave)
        if not ruta.is_file():
            return None
        datos = leer_json(ruta)
        return datos["reporte"] if datos.get("clave") == clave else None

    def guardar(self, clave, reporte, corrida):
        if reporte["estado_evaluacion"] in estados.NO_ATRIBUIBLES:
            return  # un fallo de infraestructura se puede reintentar
        self.carpeta.mkdir(parents=True, exist_ok=True)
        ruta = self._ruta(clave)
        if not ruta.exists():
            escribir_json(ruta, {"clave": clave, "corrida": corrida, "reporte": reporte})


# -- estado de un episodio -----------------------------------------------------------------------------
def estado_episodio(entradas, referencia, tolerancia):
    """El estado del episodio a partir de sus evaluaciones (en orden): la mejor (empieza en la referencia), cuántas
    propuestas seguidas no la mejoraron (un fallo de infraestructura no cuenta) y cuántos pasos lleva."""
    mejor, sin_mejora = referencia, 0
    for e in entradas:
        if e["estado_evaluacion"] == estados.CORRECTA and e.get("costo_total") is not None \
                and e["costo_total"] < mejor["costo_total"] - tolerancia:
            mejor, sin_mejora = e, 0
        elif e["estado_evaluacion"] not in estados.NO_ATRIBUIBLES:
            sin_mejora += 1
    return {"mejor": mejor, "sin_mejora": sin_mejora, "pasos": len(entradas)}
