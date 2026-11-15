"""Ciclo agente Greedy <-> evaluador (solución Dijkstra + Greedy).

    evaluador     --reporte de ciclos de cada PE (simulador)--> pesos de los PEs
    agente Greedy --camino por los PEs de menor peso----------> PEs de la configuración

El evaluador, por cada operación de operaciones.c:

    skills 01-03  cgra.json, operaciones.ll + memoria.bin (clang-18) y grafos .dot (opt-18)
    Dijkstra      sustituye los skills 04-07: grafo, compatibilidad por tipo de PE, mapeo y ciclos
    skills 08-09  PE*.txt y su validación (E03)
    simulador     cgra/src/run_cgra.py: reporte de ciclos (por PE) y memoria final
    costos        C_A, C_L, C_R y C con las mismas ecuaciones de la solución RL (contrato.py)

Greedy (dijkstra/agente.py): el grafo es la malla (por defecto 32x32, malla en
config/agente.json); cada vértice es un PE y cada arista la conexión con un vecino norte,
sur, este u oeste. Peso de un PE = w_tipo · (1 + NOPs/ciclos), con w_tipo de los pesos
w1-w4 de costos.json y NOPs y ciclos del reporte de ciclos. La malla se simula una vez;
después el camino empieza en PE00 y elige el siguiente PE con una política epsilon-greedy
(epsilon y semilla en config/agente.json): con probabilidad epsilon un vecino sin visitar
al azar, si no el de menor peso. Termina cuando el PE actual no tiene vecinos sin visitar.

    python3 agente_greedy.py [--salida resultados/greedy]
    python3 agente_greedy.py --continuar --salida resultados/greedy   # reusa la simulación guardada

Resultados en --salida:

    evaluacion_malla.json  la evaluación de la malla: C, costos por operación y NOPs por PE
    pesos.json             el peso de cada PE y los datos con que se calculó
    camino.json            los PEs del camino en orden, con su tipo y peso, el mapa ('.' = fuera) y su
                           valuación con la función de costo (C, C_A, C_L, C_R), sin volver a simular
    resumen.json           malla, inicio, PEs del camino por tipo, peso total, C del camino y tiempo
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dijkstra import CARPETA_PROGRAMAS, RUTA_CGRA_JSON, leer_cgra
from dijkstra import operaciones as operaciones_del_c
from dijkstra.agente import Agente, Parametros, RUTA_PARAMETROS, medir_en_paralelo
from dijkstra.compilar import ErrorCompilacion, preparar
from dijkstra.contrato import RUTA_COSTOS, Costos
from dijkstra.evaluador import Evaluacion, EvaluacionOperacion

ARCHIVO_EVALUACION = "evaluacion_malla.json"


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _escribir_json(ruta, datos):
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def leer_evaluacion(ruta, configuracion):
    """La evaluación guardada de la malla, si es de esta misma configuración; si no, ``None``."""
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if datos.get("configuracion") != configuracion:
        return None
    e = datos["evaluacion"]
    e["operaciones"] = {op: EvaluacionOperacion(**o) for op, o in e["operaciones"].items()}
    return Evaluacion(**e)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", default=Path("resultados/greedy"), type=Path,
                        help="Carpeta de resultados (por defecto resultados/greedy)")
    parser.add_argument("--parametros", default=RUTA_PARAMETROS, help="Parámetros del agente (config/agente.json)")
    parser.add_argument("--costos", default=RUTA_COSTOS, help="Ecuaciones y pesos w1-w4 (config/costos.json)")
    parser.add_argument("--cgra", default=RUTA_CGRA_JSON, help="Esquema de la CGRA (cgra/src/cgra.json)")
    parser.add_argument("--procesos", type=int, help="Procesos para simular las operaciones en paralelo")
    parser.add_argument("--forzar-compilacion", action="store_true", help="Repetir los skills 02-03")
    parser.add_argument("--continuar", action="store_true",
                        help="Reusar la simulación de la malla guardada en --salida (si es la misma malla)")
    args = parser.parse_args(argv)

    if (args.salida / "camino.json").exists() and not args.continuar:
        parser.exit(1, f"{args.salida} ya tiene resultados; use otra carpeta con --salida o --continuar\n")
    try:
        base = leer_cgra(args.cgra)
        parametros = Parametros.cargar(args.parametros)
        costos = Costos.cargar(args.costos)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo leer la configuración: {error}\n")
    if args.procesos is not None:
        parametros.procesos = args.procesos

    operaciones = operaciones_del_c()
    try:
        compiladas = preparar(operaciones, forzar=args.forzar_compilacion)  # skills 02-03
    except ErrorCompilacion as error:
        parser.exit(1, f"{error}\n")
    print(f"Skills 01-03: {', '.join(operaciones)}"
          f" ({'compiladas: ' + ', '.join(compiladas) if compiladas else 'IR, memoria y grafos al día'})")

    args.salida.mkdir(parents=True, exist_ok=True)
    inicio = time.monotonic()
    agente = Agente(lambda c: medir_en_paralelo(c, operaciones, CARPETA_PROGRAMAS, args.costos,
                                                parametros.t_max_simulacion_s, parametros.procesos),
                    base, parametros, costos)
    malla = agente.malla()
    configuracion = agente.configuracion(malla)
    ruta_evaluacion = args.salida / ARCHIVO_EVALUACION
    evaluacion = leer_evaluacion(ruta_evaluacion, configuracion) if args.continuar else None
    if evaluacion is None:
        print(f"Simulando la malla {len(malla)}x{len(malla[0])} (Dijkstra + simulador, una vez; "
              f"en 32x32 tarda horas)...", flush=True)
        evaluacion = agente.medir(configuracion)
        _escribir_json(ruta_evaluacion, {"configuracion": configuracion, "evaluacion": evaluacion.como_dict(),
                                         "marca_tiempo": _ahora()})
    else:
        print(f"Malla {len(malla)}x{len(malla[0])}: se reusa la simulación de {ruta_evaluacion}")
    costo = "-" if evaluacion.costo is None else f"{evaluacion.costo:.6f}"
    print(f"Malla {len(malla)}x{len(malla[0])}: {evaluacion.estado}, C = {costo}")

    try:
        agente.buscar(evaluacion)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    resultado = agente.resultado()
    _escribir_json(args.salida / "pesos.json", {
        "formula": "peso = w_tipo · (1 + NOPs/ciclos); w_tipo de pesos_area (w1-w4) en costos.json; "
                   "NOPs y ciclos del reporte de ciclos, sumados sobre las operaciones",
        "pesos": [asdict(p) for p in agente.pesos.values()],
    })
    _escribir_json(args.salida / "camino.json", resultado)
    _escribir_json(args.salida / "resumen.json", {
        "id": "dijkstra-greedy",
        "actualizado": _ahora(),
        "malla": f"{len(malla)}x{len(malla[0])}",
        "estado_malla": evaluacion.estado,
        "C_malla": evaluacion.costo,
        "inicio": resultado["inicio"],
        "pes_camino": resultado["pes"],
        "por_tipo": resultado["por_tipo"],
        "peso_total": resultado["peso_total"],
        "epsilon": resultado["epsilon"],
        "semilla": resultado["semilla"],
        "exploraciones": resultado["exploraciones"],
        "C_camino": resultado["valuacion"]["C"],
        "parametros_agente": vars(parametros),
        "tiempo_total_s": round(time.monotonic() - inicio, 1),
    })
    print(f"Camino epsilon-greedy (epsilon = {resultado['epsilon']}, semilla = {resultado['semilla']}) desde "
          f"{resultado['inicio']}: {resultado['pes']} PEs {resultado['por_tipo']}, "
          f"{resultado['exploraciones']} exploraciones, peso total {resultado['peso_total']:.6g}")
    v = resultado["valuacion"]
    print(f"Valuación del camino: C = {v['C']:.6f} (C_A = {v['C_A']:g}, C_L_norm = {v['C_L_norm']:.6g}, "
          f"C_R = {v['C_R']:.3g}); malla completa: C = {costo}")
    print(f"Resultados en {args.salida}: camino.json, pesos.json, evaluacion_malla.json, resumen.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
