"""Etapa independiente: calcular costos desde reportes JSON del simulador.

No compila, mapea ni simula. Cada --reporte se indica como OPERACION=CARPETA, la
carpeta de evaluación que arma mapear_dijkstra.py: después de correr cgra/src/run_cgra.py
con --write-back tiene reporte_ciclos.txt y la memoria final en memoria.bin. Si
Dijkstra no pudo mapear o sus PE*.txt no pasaron E03, la carpeta tiene fallo.json
(fallo_evaluador). Si el simulador falló o superó el tiempo máximo, use
--fallo OPERACION=fallo_simulador[:detalle].

Reglas y costo del contrato con la solución RL (dijkstra/contrato.py):
validación previa, validación funcional elemento por elemento, C_R absoluto,
C = promedio_op(K_A·C_A/C_A_ref + K_L·C_L,op/C_L_ref,op + K_R·C_R,op) o P_fallo.
"""
import argparse
import json
from pathlib import Path

from dijkstra import CARPETA_PROGRAMAS, RUTA_CGRA_JSON, cargar_grafo, leer_cgra
from dijkstra.contrato import (CORRECTA, FALLO_EVALUADOR, FALLO_SIMULADOR, ERROR_INFRAESTRUCTURA,
                               RUTA_COSTOS, Costos, cargar_normalizadores, validar_accion)
from dijkstra.evaluador import EvaluacionOperacion, Evaluador, agregar, invalida, medir_reporte
from dijkstra.simulacion import leer_corrida

ARCHIVO_FALLO = "fallo.json"

FALLOS_EXTERNOS = (FALLO_EVALUADOR, FALLO_SIMULADOR, ERROR_INFRAESTRUCTURA)


def _normalizadores(normalizadores, operaciones, formato, referencia, costos):
    """Los normalizadores de la búsqueda. Si falta una operación y esta es la referencia
    correcta, se usa lo que midió (como el RL cuando no hay normalizadores fijos)."""
    fijos = normalizadores or {"C_A_ref": None, "operaciones": {}}
    faltan = [op for op in operaciones if op not in fijos["operaciones"]]
    if not faltan:
        return fijos
    es_referencia = formato.malla == [f.split() for f in referencia]
    if not es_referencia or any(operaciones[op].estado != CORRECTA for op in faltan):
        raise ValueError(f"faltan C_L_ref de {', '.join(faltan)}: evalúe primero la referencia y pase "
                         "--normalizadores con el normalizacion.json que escribe agente_greedy.py")
    c_a_ref = fijos.get("C_A_ref") or costos.area(formato)
    por_op = dict(fijos["operaciones"])
    por_op.update({op: {"C_A_ref": c_a_ref, "C_L_ref": operaciones[op].C_L, "medido_con": "Dijkstra"}
                   for op in faltan})
    return {**fijos, "C_A_ref": c_a_ref, "operaciones": por_op}


def evaluar_reportes(reportes, costos, fallos=None, configuracion=None, normalizadores="congelados",
                     carpeta_programas=CARPETA_PROGRAMAS, esquema=RUTA_CGRA_JSON):
    """``reportes``: {operación: carpeta de evaluación (o su fallo.json)}; ``fallos``: {operación: (estado, detalle)};
    ``configuracion``: el cgra.json evaluado (hace falta si ningún reporte lo trae)."""
    fallos = dict(fallos or {})
    if normalizadores == "congelados":
        normalizadores = cargar_normalizadores(costos)
    tipos_esquema = leer_cgra(esquema)["tipos_pe"]
    operaciones, formato = {}, None
    carpeta_programas = Path(carpeta_programas)

    def misma(datos_configuracion):
        nonlocal formato, configuracion
        actual = Evaluador.formato(datos_configuracion)
        if formato is not None and Evaluador._clave(actual) != Evaluador._clave(formato):
            raise ValueError("Los reportes deben corresponder a la misma configuración CGRA")
        formato, configuracion = actual, datos_configuracion

    if configuracion is not None:
        misma(configuracion)
    for operacion, ruta in reportes.items():
        ruta = Path(ruta)
        fallo = ruta / ARCHIVO_FALLO if ruta.is_dir() else ruta
        if fallo.is_file():  # fallo.json de mapear_dijkstra.py
            datos = json.loads(fallo.read_text(encoding="utf-8"))
            if datos["estado"] not in FALLOS_EXTERNOS:
                raise ValueError(f"{ruta}: estado '{datos['estado']}' no es un fallo de etapa")
            if datos.get("configuracion"):
                misma(datos["configuracion"])
            operaciones[operacion] = EvaluacionOperacion(operacion, datos["estado"], datos.get("motivo", ""),
                                                         errores=datos.get("errores", []))
            continue
        ruta_memoria = carpeta_programas / operacion / "memoria.bin"
        datos = leer_corrida(ruta, ruta_memoria)
        misma(datos["configuracion"])
        grafo = cargar_grafo(operacion, carpeta_programas / operacion)
        regiones = sorted({n.banco for n in grafo.operaciones if n.op == "store"})
        operaciones[operacion] = medir_reporte(operacion, datos, grafo, regiones, costos)
    for operacion, (estado, detalle) in fallos.items():
        if operacion in operaciones:
            raise ValueError(f"{operacion} tiene reporte y fallo a la vez")
        if estado not in FALLOS_EXTERNOS:
            raise ValueError(f"--fallo {operacion}: estado debe ser uno de {', '.join(FALLOS_EXTERNOS)}")
        operaciones[operacion] = EvaluacionOperacion(
            operacion, estado, detalle, errores=[{"skill": "E04" if estado == FALLO_SIMULADOR else "E02",
                                                  "regla": estado, "detalle": detalle}])
    if formato is None:
        raise ValueError("Debe proporcionar al menos un reporte, o --cgra con la configuración")
    if not operaciones:
        raise ValueError("Debe proporcionar al menos un reporte o fallo")
    motivos = validar_accion(configuracion, formato, costos.restricciones, tipos_esquema)
    if motivos:
        evaluacion = invalida(motivos, operaciones, costos, formato.mapa())
    else:
        simuladas = any(e.C_L is not None for e in operaciones.values())
        if simuladas:
            normalizadores = _normalizadores(normalizadores, operaciones, formato, costos.referencia, costos)
        evaluacion = agregar(formato, operaciones, costos, normalizadores)
    return dict(configuracion=configuracion, evaluacion=evaluacion.como_dict(),
                criterio=costos.criterio(evaluacion.normalizadores))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reporte", action="append", default=[], metavar="OPERACION=CARPETA")
    parser.add_argument("--fallo", action="append", default=[], metavar="OPERACION=ESTADO[:DETALLE]",
                        help=f"Operación que no llegó a reporte: {', '.join(FALLOS_EXTERNOS)}")
    parser.add_argument("--cgra", type=Path, help="Configuración evaluada (necesaria si solo hay --fallo)")
    parser.add_argument("--costos", default=RUTA_COSTOS)
    parser.add_argument("--normalizadores", type=Path,
                        help="normalizacion.json de la búsqueda (por defecto, el congelado de costos.json)")
    parser.add_argument("--salida", required=True, type=Path)
    args = parser.parse_args()
    try:
        reportes, fallos = {}, {}
        for entrada in args.reporte:
            operacion, ruta = entrada.split("=", 1)
            if operacion in reportes:
                raise ValueError(f"Reporte duplicado para {operacion}")
            reportes[operacion] = ruta
        for entrada in args.fallo:
            operacion, resto = entrada.split("=", 1)
            estado, _, detalle = resto.partition(":")
            fallos[operacion] = (estado, detalle or estado)
        costos = Costos.cargar(args.costos)
        normalizadores = "congelados"
        if args.normalizadores:
            normalizadores = json.loads(args.normalizadores.read_text(encoding="utf-8"))
        configuracion = json.loads(args.cgra.read_text(encoding="utf-8")) if args.cgra else None
        resultado = evaluar_reportes(reportes, costos, fallos, configuracion, normalizadores)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"No se pudo evaluar: {error}\n")
    e = resultado["evaluacion"]
    m = e["metricas"]
    print(f"{e['estado']}: C_A={m.get('C_A')} C_L_norm={m.get('C_L_norm')} C_R={m.get('C_R')} C={e['costo']}")
    print(f"Evaluación guardada en {args.salida}")


if __name__ == "__main__":
    main()
