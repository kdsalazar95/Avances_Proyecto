"""Medición: conecta el evaluador que ya estaba hecho (Skills 01–09 + simulador Python) y mide.

    cgra.json del agente + .c
      E01  recepción y carpeta aislada
      E02  Skills 01–09 generan los PE*.txt          (compilador.py, LLM en otra sesión)
      E03  ¿están bien los .txt?  no -> reintento con los errores (N_corr) -> "Reportar"
      E04  correr el simulador (cgra/src/run_cgra.py) -> reporte por ciclos + memoria
      E05  validación funcional elemento por elemento
      E06  C_A (del cgra.json), C_L (ciclos), C_R y costo
      E07  recompensa
      E08  reporte -> agente

Interfaz que usa el ciclo (A08)::

    evaluar(dir_candidata, id_evaluacion, instancia, corrida, normalizadores,
            mejor_costo_anterior, carpeta_eval, compilador) -> reporte

``normalizadores=None`` indica la evaluación de la referencia: sus C_A y C_L se
usan como C_A,ref y C_L,ref (costo de referencia = K_A + K_L + K_R·C_R).
"""

import shutil
import time
from pathlib import Path

from .. import estados
from ..comun import (ARCHIVO_CONFIG, ARCHIVO_MANIFEST, CARPETA_PROGRAMAS, escribir_json, hash_accion,
                     hash_conjunto, hashes_paquete, leer_json, relativa)
from . import funcional, metricas, preparacion, recompensa, reporte, simulacion
from .validacion_programas import validar_programas


def accion_de(id_evaluacion, dir_candidata):
    """Identifica la acción evaluada: el cgra.json de la candidata y su hash."""
    dir_candidata = Path(dir_candidata)
    hashes = {k: v for k, v in hashes_paquete(dir_candidata).items() if k == ARCHIVO_CONFIG}
    return {
        "id_candidata": id_evaluacion,
        "directorio": relativa(dir_candidata),
        "configuracion": relativa(dir_candidata / ARCHIVO_CONFIG),
        "hashes": hashes,
        "hash_accion": hash_accion(hashes),
    }


def _texto_error(e):
    lugar = " ".join(str(x) for x in (e.get("archivo"), f"ciclo {e['ciclo']}" if e.get("ciclo") is not None
                                      else None) if x)
    return f"[{e.get('regla')}] {lugar} {e.get('detalle')}".replace("  ", " ")


def compilar_con_skills(compilador, carpeta_eval, instancia, n_corr):
    """E02 + E03 con reintentos. Devuelve ``(estado | None, errores, info_programas)``.

    ``estado`` None significa que hay PE*.txt válidos en ``carpeta_eval``.
    """
    carpeta_eval = Path(carpeta_eval)
    ruta_cgra = carpeta_eval / ARCHIVO_CONFIG
    errores_previos, historial = None, []
    for intento in range(1, n_corr + 2):
        destino = carpeta_eval / "skills" / f"intento-{intento}"
        resultado = compilador.compilar(ruta_cgra, instancia, destino, errores_previos)
        if resultado["infraestructura"]:
            error = {"skill": "E02", "regla": "infraestructura", "detalle": resultado["detalle"]}
            return estados.ERROR_INFRAESTRUCTURA, historial + [error], {"intentos": intento}
        if not resultado["ok"]:
            # "Reportar": los skills mismos no pudieron generar los programas.
            error = {"skill": "E02", "regla": "skills_reportan", "detalle": resultado["detalle"],
                     "intento": intento}
            return estados.FALLO_EVALUADOR, historial + [error], {"intentos": intento}
        manifest_ruta = destino / ARCHIVO_MANIFEST
        try:
            manifest = leer_json(manifest_ruta) if manifest_ruta.is_file() else {}
        except ValueError:
            manifest = {}
        e03 = validar_programas(ruta_cgra, destino / CARPETA_PROGRAMAS, carpeta_eval / "memoria_inicial.bin",
                                manifest, instancia["regiones_salida"])
        if e03.get("fallo_infraestructura"):
            return estados.ERROR_INFRAESTRUCTURA, historial + e03["errores"], {"intentos": intento}
        if e03["valida"]:
            shutil.copytree(destino / CARPETA_PROGRAMAS, carpeta_eval / CARPETA_PROGRAMAS)
            if manifest_ruta.is_file():
                shutil.copy2(manifest_ruta, carpeta_eval / ARCHIVO_MANIFEST)
            hashes = {k: v for k, v in hashes_paquete(carpeta_eval).items() if k.startswith(CARPETA_PROGRAMAS)}
            return None, historial, {"intentos": intento, "directorio": relativa(carpeta_eval / CARPETA_PROGRAMAS),
                                     "hash_programas": hash_conjunto(hashes), "archivos": len(hashes),
                                     "ciclos_programa": e03.get("ciclos_programa")}
        for e in e03["errores"]:
            historial.append({**e, "intento": intento})
        errores_previos = [_texto_error(e) for e in e03["errores"]]
    historial.append({"skill": "E03", "regla": "reportar",
                      "detalle": f"los .txt siguen con errores después de {n_corr} intentos de corrección"})
    return estados.FALLO_EVALUADOR, historial, {"intentos": n_corr + 1}


def evaluar(dir_candidata, id_evaluacion, instancia, corrida, normalizadores, mejor_costo_anterior, carpeta_eval,
            compilador):
    inicio = time.monotonic()
    p_fallo = corrida["penalizacion"]["P_fallo"]
    es_referencia = normalizadores is None
    accion = accion_de(id_evaluacion, dir_candidata)
    errores, avisos, programas = [], [], None

    def cerrar(estado, metricas_=None, costo=None, validacion=None, normalizadores_=normalizadores):
        costo_final = recompensa.costo_reportado(estado, costo, p_fallo)
        if es_referencia and estado == estados.CORRECTA:
            r, detalle = 0.0, {"formula": "referencia: C_best_0 = C_ref", "C_best_prev": None, "P_fallo": p_fallo}
        else:
            r, detalle = recompensa.recompensa(estado, costo_final, mejor_costo_anterior, p_fallo)
        doc = reporte.construir(
            id_evaluacion, accion, estado, metricas_, costo_final, r, validacion, errores,
            round(time.monotonic() - inicio, 3), reporte.parametros_usados(corrida, normalizadores_), detalle, avisos,
        )
        doc["programas_generados"] = programas
        doc["es_referencia"] = es_referencia
        if Path(carpeta_eval).is_dir():
            escribir_json(Path(carpeta_eval) / "reporte.json", doc)
        return doc

    # E01: carpeta aislada con el cgra.json del agente y la memoria inicial.
    e01 = preparacion.recibir(dir_candidata, instancia, carpeta_eval, id_evaluacion)
    if not e01["listo"]:
        errores.extend(e01["errores"])
        return cerrar(estados.ERROR_INFRAESTRUCTURA)

    # E02 + E03: Skills 01–09 generan los PE*.txt; se validan y, si fallan, se reintenta.
    n_corr = corrida["compilador"].get("N_corr", 2)
    estado, errores_skills, programas = compilar_con_skills(compilador, carpeta_eval, instancia, n_corr)
    errores.extend(errores_skills)
    if estado is not None:
        return cerrar(estado)

    # E04: simulador existente, proceso aparte con timeout.
    e04 = simulacion.ejecutar(carpeta_eval, corrida["limites"]["t_max_simulacion_s"])
    avisos.extend(e04.get("avisos", []))
    if not e04["ok"]:
        errores.extend(e04["errores"])
        return cerrar(estados.FALLO_SIMULADOR, {"ciclos": None, "nops": None})

    # E05: validación funcional contra el esperado independiente.
    esperado = leer_json(instancia["esperado"])
    if esperado.get("hash_memoria_inicial") != instancia["hash_memoria_inicial"]:
        errores.append({"skill": "E05", "regla": "esperado_desligado",
                        "detalle": "esperado.json no corresponde a la memoria inicial de la instancia"})
        return cerrar(estados.ERROR_INFRAESTRUCTURA)
    e05 = funcional.validar_memoria(e04["ruta_memoria"], esperado["regiones"], corrida["tolerancias"])
    for texto in e05["errores_estructurales"]:
        errores.append({"skill": "E05", "regla": "estructura", "detalle": texto})
    if e05["elementos_fallidos"]:
        errores.append({"skill": "E05", "regla": "fuera_de_tolerancia",
                        "detalle": f"{e05['elementos_fallidos']} de {e05['total_elementos']} elementos fallan",
                        "primeros_fallos": e05["primeros_fallos"]})

    # E06: métricas y costo.
    ruta_cgra = Path(carpeta_eval) / ARCHIVO_CONFIG
    normalizadores_ = normalizadores
    if es_referencia:
        c_a_ref, _ = metricas.costo_area_json(ruta_cgra, corrida["pesos_area"])
        normalizadores_ = {"C_A_ref": c_a_ref, "C_L_ref": metricas.costo_latencia(
            e04["ciclos"], e04["nops"], corrida["formula_latencia"], metricas.numero_pes(ruta_cgra)),
            "referencia": id_evaluacion}
    m = metricas.calcular(ruta_cgra, e04["ciclos"], e04["nops"], e05["C_R"], corrida, normalizadores_)
    m["nops_por_pe"] = e04.get("nops_por_pe")
    validacion = {k: e05[k] for k in ("aprobada", "elementos_fallidos", "total_elementos", "primeros_fallos")}
    estado = estados.CORRECTA if e05["aprobada"] else estados.INCORRECTA
    return cerrar(estado, m, m["costo_total"], validacion, normalizadores_)
