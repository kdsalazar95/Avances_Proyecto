"""Etapa Greedy por rondas: leer evaluaciones y proponer configuraciones.

No compila, mapea, simula ni calcula costos. Sin --evaluaciones propone la
referencia (costos.json → referencia). En las siguientes invocaciones lee las
evaluaciones de evaluar_cgra.py, las registra como el RL (historial.jsonl,
optimas.json, resumen.json, con recompensa r = C_best − C), actualiza la cadena
de mejoras y exporta las vecinas pendientes de las mejores, sin pasar del
presupuesto (max_evaluaciones y paciencia de agente.json).
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from dijkstra import RUTA_CGRA_JSON, leer_cgra
from dijkstra.agente import Agente, Parametros, RUTA_PARAMETROS, clave
from dijkstra.contrato import (CORRECTA, FORMULA_RECOMPENSA, NO_ATRIBUIBLES, PENALIZADOS, RUTA_COSTOS, Costos,
                               actualizar_optimas, hash_configuracion, recompensa, verificar_penalizacion)
from dijkstra.evaluador import Evaluador

CAMPOS_METRICAS = ("C_A", "C_L", "C_R", "C_A_norm", "C_L_norm", "ciclos", "nops", "fraccion_nops", "capacidades")


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sin_normalizadores(criterio):
    return {k: v for k, v in criterio.items() if k != "normalizadores"}


def entrada_historial(dato, id_evaluacion, iteracion, ronda, r, c_best_prev, costos, es_referencia):
    """Una línea de historial.jsonl con los campos del reporte E08 del RL (rl/formatos.md)."""
    e = dato["evaluacion"]
    configuracion = dato["configuracion"]
    h = hash_configuracion(configuracion)
    por_operacion = {
        op: {"estado": o["estado"], "costo": o["costo"], "ciclos": o["ciclos"], "nops": o["nops"],
             "fraccion_nops": o["fraccion_nops"], "C_L": o["C_L"], "nops_por_pe": None,
             "C_L_norm": o["C_L_norm"], "C_R": o["C_R"], "intentos_skills": None, "modo": o["modo"]}
        for op, o in e["operaciones"].items()}
    validaciones = {op: o["validacion"] for op, o in e["operaciones"].items()}
    return {
        "id_evaluacion": id_evaluacion,
        "accion": {"id_candidata": id_evaluacion, "directorio": None, "configuracion": configuracion,
                   "hashes": {"cgra.json": h}, "hash_accion": h},
        "estado_evaluacion": e["estado"],
        "metricas": {k: e["metricas"].get(k) for k in CAMPOS_METRICAS},
        "costo_total": e["costo"],
        "costo_penalizado": e["estado"] in PENALIZADOS,
        "recompensa": r,
        "detalle_recompensa": {"formula": "referencia: C_best_0 = C_ref" if es_referencia else FORMULA_RECOMPENSA,
                               "C_best_prev": c_best_prev, "P_fallo": costos.p_fallo},
        "validacion_funcional": {"aprobada": all((v or {}).get("aprobada") for v in validaciones.values()),
                                 "por_operacion": validaciones},
        "errores": e["errores"],
        "avisos": [],
        "tiempo_evaluacion_s": None,
        "parametros": costos.parametros_usados(e["normalizadores"]),
        "programas_generados": None,
        "por_operacion": por_operacion,
        "es_referencia": es_referencia,
        "iteracion": iteracion,
        "ronda": ronda,
        "marca_tiempo": _ahora(),
        "duplicada_de": None,
        "malla": e["mapa"],
    }


def reconstruir(historial, tolerancia):
    """Óptimas, C_min y evaluaciones seguidas sin mejora (rl/agente/memoria.py)."""
    optimas, c_min, sin_mejora = [], None, 0
    for entrada in historial:
        optimas, c_min, mejora = actualizar_optimas(optimas, entrada, tolerancia)
        if not entrada["es_referencia"] and entrada["estado_evaluacion"] not in NO_ATRIBUIBLES:
            sin_mejora = 0 if mejora else sin_mejora + 1
    return optimas, c_min, sin_mejora


def avanzar(base, parametros, estado, nuevas, costos=None):
    """Registra ``nuevas`` (salidas de evaluar_cgra.py) y propone la siguiente ronda.

    Devuelve ``(estado, propuestas, entradas_nuevas_del_historial)``."""
    costos = costos or Costos.cargar()
    agente = Agente(None, base, parametros, costos=costos)
    referencia = agente.referencia()
    clave_ref = clave(referencia)
    evaluaciones = dict(estado.get("evaluaciones", {}))
    historial = list(estado.get("historial", []))
    criterio = estado.get("criterio")
    normalizadores = estado.get("normalizadores")
    ronda = estado.get("ronda", 0)
    esperado = _sin_normalizadores(costos.criterio(None))
    optimas, c_min, _ = reconstruir(historial, costos.tolerancia_empate)
    c_best_prev = c_min  # todo el lote se mide contra el mejor anterior al lote
    nuevas_entradas = []
    for dato in nuevas:
        if _sin_normalizadores(dato["criterio"]) != esperado:
            raise ValueError("La evaluación usa pesos, K, tolerancias o fórmulas distintos de costos.json")
        if criterio is not None and _sin_normalizadores(dato["criterio"]) != criterio:
            raise ValueError("No mezcle evaluaciones con pesos o medidas de costo diferentes")
        criterio = _sin_normalizadores(dato["criterio"])
        usados = dato["evaluacion"]["normalizadores"]
        if usados is not None:
            if normalizadores is not None and usados != normalizadores:
                raise ValueError("La evaluación usa otros normalizadores; pase --normalizadores "
                                 "con el normalizacion.json de esta búsqueda")
            normalizadores = normalizadores or usados
        formato = Evaluador.formato(dato["configuracion"])
        malla = tuple(tuple(f) for f in formato.malla)
        # Greedy no explora cambios de memoria, ISA ni ancho vectorial.
        canonica = Evaluador.formato(agente.configuracion(malla))
        if Evaluador._clave(formato) != Evaluador._clave(canonica):
            raise ValueError("La evaluación usa hardware distinto del espacio de diseño")
        k = clave(malla)
        if not historial and not nuevas_entradas and k != clave_ref:
            raise ValueError("La primera evaluación debe ser la de la referencia "
                             f"({' / '.join(costos.referencia)}): C_best,0 = C_ref")
        if k in evaluaciones:
            if evaluaciones[k] != dato:
                raise ValueError(f"La configuración {k} ya tiene otra evaluación; use otra carpeta de salida")
            continue
        e = dato["evaluacion"]
        if e["costo"] is None and e["estado"] not in NO_ATRIBUIBLES:
            raise ValueError(f"{k}: solo error_infraestructura puede no tener costo")
        if e["costo"] is not None and not math.isfinite(e["costo"]):
            raise ValueError("Greedy requiere costos finitos")
        es_referencia = k == clave_ref
        if es_referencia:
            r = 0.0 if e["estado"] == CORRECTA else recompensa(e["estado"], e["costo"], None, costos.p_fallo)
        else:
            r = recompensa(e["estado"], e["costo"], c_best_prev, costos.p_fallo)
        iteracion = len(historial)
        id_evaluacion = "referencia" if es_referencia else f"it-{iteracion:04d}"
        entrada = entrada_historial(dato, id_evaluacion, iteracion, ronda, r,
                                    None if es_referencia else c_best_prev, costos, es_referencia)
        historial.append(entrada)
        nuevas_entradas.append(entrada)
        evaluaciones[k] = dato
        if es_referencia and e["estado"] == CORRECTA:
            c_best_prev = e["costo"]
    if nuevas_entradas:
        ronda += 1

    # Cadena de mejoras (regla de Greedy): cada válida con C ≤ el último de la cadena entra.
    cadena, umbral = [], float("inf")
    for k, dato in evaluaciones.items():
        e = dato["evaluacion"]
        if e["estado"] == CORRECTA and e["costo"] <= umbral:
            cadena.append(k)
            umbral = e["costo"]

    optimas, c_min, sin_mejora = reconstruir(historial, costos.tolerancia_empate)
    hechas = sum(not h["es_referencia"] for h in historial)
    ref = next((h for h in historial if h["es_referencia"]), None)
    terminada = None
    if ref is not None and ref["estado_evaluacion"] != CORRECTA:
        terminada = "detenida: referencia no válida"
    elif parametros.max_evaluaciones is not None and hechas >= parametros.max_evaluaciones:
        terminada = "terminada: max_evaluaciones"
    elif parametros.paciencia is not None and sin_mejora >= parametros.paciencia:
        terminada = "terminada: paciencia"

    propuestas = [tuple(tuple(f.split()) for f in c["malla"]) for c in estado.get("pendientes", [])]
    if not evaluaciones:
        propuestas = [referencia]
    elif terminada is None:
        for k, dato in evaluaciones.items():
            e = dato["evaluacion"]
            if e["estado"] == CORRECTA and e["costo"] == umbral:
                f = Evaluador.formato(dato["configuracion"])
                propuestas.extend(agente.vecinas(tuple(tuple(fila) for fila in f.malla)))
    propuestas = [m for m in dict.fromkeys(propuestas)
                  if clave(m) not in evaluaciones and not agente.motivo_invalida(m)]
    if terminada is not None:
        propuestas = []
    elif evaluaciones and parametros.max_evaluaciones is not None:
        propuestas = propuestas[:max(parametros.max_evaluaciones - hechas, 0)]
    if evaluaciones and not propuestas and terminada is None:
        terminada = "terminada: sin vecinas por explorar"

    configuraciones = [agente.configuracion(m) for m in propuestas]
    estado = dict(evaluaciones=evaluaciones, historial=historial, criterio=criterio, normalizadores=normalizadores,
                  cadena=cadena, ronda=ronda, base=base, parametros=vars(parametros), pendientes=configuraciones,
                  estado_corrida=terminada or "en_curso", optimas=optimas, C_min=c_min,
                  iteraciones_sin_mejora=sin_mejora)
    return estado, configuraciones, nuevas_entradas


def resumen(estado, costos):
    """``resumen.json`` con los campos del RL (rl/agente/ciclo.py)."""
    entradas = estado["historial"]
    agente = [e for e in entradas if not e["es_referencia"]]
    correctos = [e["costo_total"] for e in entradas if e["estado_evaluacion"] == CORRECTA]
    validas = [e for e in agente if e["estado_evaluacion"] != "accion_invalida"]
    conteo = {}
    for e in entradas:
        conteo[e["estado_evaluacion"]] = conteo.get(e["estado_evaluacion"], 0) + 1
    referencia = next((e for e in entradas if e["es_referencia"]), None)
    return {
        "id": "dijkstra-greedy",
        "estado": estado["estado_corrida"],
        "actualizado": _ahora(),
        "C_min": estado["C_min"],
        "optimas": [o["id_candidata"] for o in estado["optimas"]],
        "evaluaciones": len(entradas),
        "evaluaciones_agente": len(agente),
        "iteraciones_sin_mejora": estado["iteraciones_sin_mejora"],
        "conteo_por_estado": conteo,
        "porcentaje_validas": round(100 * len(validas) / len(agente), 1) if agente else None,
        "porcentaje_correctas": round(100 * sum(e["estado_evaluacion"] == CORRECTA for e in agente)
                                      / len(agente), 1) if agente else None,
        "evolucion_costo": [{"id": e["id_evaluacion"], "estado": e["estado_evaluacion"],
                             "costo_total": e["costo_total"], "recompensa": e["recompensa"]} for e in entradas],
        "tiempo_evaluacion_total_s": None,
        "normalizadores": estado["normalizadores"],
        "referencia_medida": None if referencia is None else {
            op: {"ciclos": d["ciclos"], "C_L": d["C_L"]} for op, d in referencia["por_operacion"].items()},
        "avisos_penalizacion": verificar_penalizacion(costos.p_fallo, correctos),
        "nota": "Óptimas encontradas dentro del presupuesto; no garantiza el óptimo global.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgra", default=RUTA_CGRA_JSON, help="Esquema de la CGRA (por defecto cgra/src/cgra.json)")
    parser.add_argument("--parametros", default=RUTA_PARAMETROS)
    parser.add_argument("--costos", default=RUTA_COSTOS)
    parser.add_argument("--lado-maximo", type=int)
    parser.add_argument("--evaluaciones", nargs="*", default=[], type=Path)
    parser.add_argument("--salida", default=Path("resultados/greedy"), type=Path)
    args = parser.parse_args()
    try:
        base = leer_cgra(args.cgra)
        parametros = Parametros.cargar(args.parametros)
        costos = Costos.cargar(args.costos)
        if args.lado_maximo is not None:
            parametros.lado_maximo = args.lado_maximo
        ruta_estado = args.salida / "estado.json"
        estado = json.loads(ruta_estado.read_text()) if ruta_estado.exists() else {}
        if estado and (estado["base"] != base or estado["parametros"] != vars(parametros)):
            raise ValueError("Cambió el espacio de diseño o parámetros; use otra carpeta de salida")
        nuevas = [json.loads(p.read_text(encoding="utf-8")) for p in args.evaluaciones]
        estado, propuestas, entradas = avanzar(base, parametros, estado, nuevas, costos)
        args.salida.mkdir(parents=True, exist_ok=True)
        ruta_estado.write_text(json.dumps(estado, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        with (args.salida / "historial.jsonl").open("a", encoding="utf-8") as historial:
            for entrada in entradas:  # solo se agregan líneas
                historial.write(json.dumps(entrada, ensure_ascii=False, allow_nan=False) + "\n")
        (args.salida / "optimas.json").write_text(json.dumps(
            {"C_min": estado["C_min"], "optimas": estado["optimas"],
             "tolerancia_empate_costo": costos.tolerancia_empate}, ensure_ascii=False, indent=2) + "\n")
        (args.salida / "resumen.json").write_text(json.dumps(resumen(estado, costos), ensure_ascii=False,
                                                             indent=2) + "\n")
        if estado["normalizadores"] is not None:
            (args.salida / "normalizacion.json").write_text(json.dumps(
                estado["normalizadores"], ensure_ascii=False, indent=2) + "\n")
        carpeta = args.salida / "propuestas"
        carpeta.mkdir(exist_ok=True)
        for viejo in carpeta.glob("config_*.json"):
            viejo.unlink()
        for i, configuracion in enumerate(propuestas, 1):
            (carpeta / f"config_{i:04d}.json").write_text(json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n")
        cadena = [estado["evaluaciones"][k] for k in estado["cadena"]]
        (args.salida / "cadena.json").write_text(json.dumps(cadena, ensure_ascii=False, indent=2) + "\n")
        if cadena:
            (args.salida / "mejor_cgra.json").write_text(json.dumps(cadena[-1]["configuracion"], ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo avanzar: {error}\n")
    print(f"{len(propuestas)} propuestas en {carpeta}; {len(cadena)} pasos en la cadena; "
          f"estado: {estado['estado_corrida']}.")
    if propuestas:
        print("Ejecute por separado Dijkstra, simulador y evaluador; luego entregue sus evaluaciones a Greedy.")


if __name__ == "__main__":
    main()
