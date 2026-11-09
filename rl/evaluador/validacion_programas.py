"""E03 · ¿Están bien los .txt? Valida los PE*.txt que generaron los Skills 01–09.

1. Reutiliza el cargador real (``FormatoCGRA`` + ``Cargador`` + ``Programa``):
   sintaxis, ISA por tipo de PE, ciclos alineados, SEND/RECV y VSEND/VRECV
   emparejados con su vecino, regiones y rangos de LD/ST/VLD/VST, memoria que
   cabe en ``palabras_memoria``.
2. Añade comprobaciones deterministas que el cargador no garantiza
   (``cgra/agent_skills/09-validate-schedule.md``):
   - conjunto exacto de PE*.txt (sin sobrantes);
   - nombres de registro existentes y de la clase correcta (escalar/vectorial);
   - registros definidos antes de usarse (incluye fuentes de ST/SEND);
   - cobertura de las regiones de salida: cada posición escrita exactamente una vez;
   - encabezado ``// PE type: ... (L)`` coherente con el ``cgra.json`` si existe.
"""

import re
from pathlib import Path

from ..comun import usar_simulador

usar_simulador()
from cargador import Cargador  # noqa: E402
from cgra import CGRA  # noqa: E402
from formato_cgra import FormatoCGRA  # noqa: E402
from isa import ISA  # noqa: E402

LECTURAS_ESCALARES = {"add": ("src1", "src2"), "sub": ("src1", "src2"), "mul": ("src1", "src2"),
                      "div": ("src1", "src2"), "st": ("src",), "send": ("src",), "vbcast": ("src",)}
ESCRITURAS_ESCALARES = {"mov", "add", "sub", "mul", "div", "ld", "recv", "vredsum"}
LECTURAS_VECTORIALES = {"vadd": ("src1", "src2"), "vsub": ("src1", "src2"), "vmul": ("src1", "src2"),
                        "vdiv": ("src1", "src2"), "vst": ("src",), "vredsum": ("src",), "vsend": ("src",)}
ESCRITURAS_VECTORIALES = {"vld", "vadd", "vsub", "vmul", "vdiv", "vbcast", "vrecv"}

PATRON_PE = re.compile(r"\b(PE\d+)\b")
PATRON_CICLO = re.compile(r"[Cc]iclo (\d+)")
PATRON_TIPO = re.compile(r"^//\s*PE type:.*\((\w)\)", re.M)


def _error(regla, detalle, archivo=None, ciclo=None):
    return {"skill": "E03", "archivo": archivo, "ciclo": ciclo, "regla": regla, "detalle": detalle}


def _desde_mensaje(regla, mensaje):
    """Convierte un mensaje del cargador en errores estructurados (uno por línea)."""
    errores = []
    for linea in str(mensaje).splitlines():
        linea = linea.strip()
        if not linea:
            continue
        pe = PATRON_PE.search(linea)
        ciclo = PATRON_CICLO.search(linea)
        errores.append(_error(regla, linea, f"{pe.group(1)}.txt" if pe else None,
                              int(ciclo.group(1)) if ciclo else None))
    return errores


def _registros(programa):
    """Nombres válidos, clase correcta y definición antes del uso, PE por PE."""
    errores = []
    for pe_id, instrucciones in programa.instrucciones.items():
        escalares, vectoriales = set(), set()
        for ciclo in sorted(instrucciones):
            inst = instrucciones[ciclo]
            op = inst["op"]
            for campo in LECTURAS_ESCALARES.get(op, ()):
                nombre = inst[campo]
                if nombre not in ISA.REGISTROS:
                    errores.append(_error("registro_inexistente", f"{op.upper()} usa '{nombre}', que no es un "
                                          f"registro escalar ({', '.join(ISA.REGISTROS)})", f"{pe_id}.txt", ciclo))
                elif nombre not in escalares:
                    errores.append(_error("registro_sin_definir", f"{op.upper()} lee '{nombre}' antes de escribirlo",
                                          f"{pe_id}.txt", ciclo))
            for campo in LECTURAS_VECTORIALES.get(op, ()):
                nombre = inst[campo]
                if nombre not in ISA.VECTORIALES:
                    errores.append(_error("registro_inexistente", f"{op.upper()} usa '{nombre}', que no es un "
                                          "registro vectorial (v0..v7)", f"{pe_id}.txt", ciclo))
                elif nombre not in vectoriales:
                    errores.append(_error("registro_sin_definir", f"{op.upper()} lee '{nombre}' antes de escribirlo",
                                          f"{pe_id}.txt", ciclo))
            if op in ESCRITURAS_ESCALARES:
                if inst["dst"] not in ISA.REGISTROS:
                    errores.append(_error("registro_inexistente", f"{op.upper()} escribe '{inst['dst']}', que no es "
                                          "un registro escalar", f"{pe_id}.txt", ciclo))
                escalares.add(inst["dst"])
            if op in ESCRITURAS_VECTORIALES:
                if inst["dst"] not in ISA.VECTORIALES:
                    errores.append(_error("registro_inexistente", f"{op.upper()} escribe '{inst['dst']}', que no es "
                                          "un registro vectorial", f"{pe_id}.txt", ciclo))
                vectoriales.add(inst["dst"])
    return errores


def _cobertura(programa, simbolos, regiones_salida, ancho_vector):
    """Cada posición de cada región de salida se escribe exactamente una vez."""
    errores = []
    escrituras = {nombre: {} for nombre in regiones_salida}
    for pe_id, instrucciones in programa.instrucciones.items():
        for ciclo, inst in sorted(instrucciones.items()):
            if inst["op"] not in {"st", "vst"} or inst["bank"] not in escrituras:
                continue
            palabras = ancho_vector if inst["op"] == "vst" else 1
            for indice in range(inst["index"], inst["index"] + palabras):
                escrituras[inst["bank"]].setdefault(indice, []).append((pe_id, ciclo))
    for nombre in regiones_salida:
        if nombre not in simbolos:
            errores.append(_error("region_salida_inexistente", f"la memoria no tiene la región '{nombre}'"))
            continue
        _, filas, columnas = simbolos[nombre]
        total = filas * columnas
        faltan = [i for i in range(total) if i not in escrituras[nombre]]
        if faltan:
            muestra = ", ".join(map(str, faltan[:10])) + (" ..." if len(faltan) > 10 else "")
            errores.append(_error("salida_sin_cubrir",
                                  f"{len(faltan)} de {total} posiciones de '{nombre}' nunca se escriben: {muestra}"))
        for indice, quienes in sorted(escrituras[nombre].items()):
            if len(quienes) > 1:
                detalle = ", ".join(f"{pe} ciclo {c}" for pe, c in quienes)
                errores.append(_error("salida_duplicada", f"'{nombre}[{indice}]' se escribe {len(quienes)} veces "
                                      f"({detalle})", f"{quienes[1][0]}.txt", quienes[1][1]))
    return errores


def _encabezados(directorio, formato, malla):
    errores = []
    for pe_id in malla.ids():
        texto = (Path(directorio) / f"{pe_id}.txt").read_text(encoding="utf-8")
        tipo = PATRON_TIPO.search(texto)
        if tipo and tipo.group(1) != formato.letra(pe_id):
            errores.append(_error("encabezado_incoherente", f"el encabezado dice tipo {tipo.group(1)} y cgra.json "
                                  f"asigna {formato.letra(pe_id)}", f"{pe_id}.txt"))
    return errores


def _manifest(manifest, formato):
    errores = []
    if manifest.get("filas") != formato.filas or manifest.get("columnas") != formato.columnas:
        errores.append(_error("manifest_incoherente", f"manifest {manifest.get('filas')}x{manifest.get('columnas')} "
                              f"y cgra.json {formato.filas}x{formato.columnas}"))
    if "malla" in manifest:
        malla = [fila.split() if isinstance(fila, str) else list(fila) for fila in manifest["malla"]]
        if malla != formato.malla:
            errores.append(_error("manifest_incoherente", "la 'malla' del manifest no coincide con la de cgra.json"))
    return errores


def validar_programas(ruta_cgra, dir_programas, ruta_memoria, manifest, regiones_salida):
    """Valida los PE*.txt que generaron los Skills 01–09 para la configuración ``ruta_cgra``.

    Devuelve ``{"valida", "errores", "fallo_infraestructura"?, "ciclos_programa"?}``.
    """
    directorio = Path(dir_programas)
    try:
        formato = FormatoCGRA.cargar(ruta_cgra)
        cgra = CGRA(formato)
    except Exception as error:  # noqa: BLE001 - A03 ya aceptó este cgra.json: es infraestructura
        return {"valida": False, "fallo_infraestructura": True,
                "errores": [_error("infraestructura", f"no se pudo crear la CGRA: {error}")]}
    if not directorio.is_dir():
        return {"valida": False, "errores": [_error("sin_programas", f"no existe {directorio.name}/")]}

    errores = []
    esperados = {f"{pe_id}.txt" for pe_id in cgra.malla.ids()}
    presentes = {ruta.name for ruta in directorio.glob("*")}
    for nombre in sorted(esperados - presentes):
        errores.append(_error("pe_faltante", f"falta {nombre} para la malla {formato.filas}x{formato.columnas}", nombre))
    for nombre in sorted(presentes - esperados):
        errores.append(_error("pe_sobrante", f"{nombre} no corresponde a ningún PE de la malla", nombre))
    if manifest:
        errores += _manifest(manifest, formato)
    if any(e["regla"] == "pe_faltante" for e in errores):
        return {"valida": False, "errores": errores}

    try:
        programa = Cargador(cgra).cargar(ruta_memoria, directorio)
    except (ValueError, FileNotFoundError) as error:
        errores += _desde_mensaje("cargador", error)
        return {"valida": False, "errores": errores}

    errores += _registros(programa)
    errores += _cobertura(programa, cgra.memoria.simbolos, regiones_salida, formato.ancho_vector)
    errores += _encabezados(directorio, formato, cgra.malla)
    return {"valida": not errores, "errores": errores, "ciclos_programa": len(programa.ciclos())}
