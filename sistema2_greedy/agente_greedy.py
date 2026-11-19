"""Ciclo agente Greedy <-> evaluador (solución Dijkstra + Greedy).

    evaluador     --reporte de ciclos de cada PE (simulador)--> pesos de los PEs
    agente Greedy --camino por los PEs de menor costo C-------> PEs de la configuración

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
(epsilon y semilla en config/agente.json): con probabilidad epsilon explora (un vecino sin visitar
al azar; este sistema no usa LLM) y si no, greedy: el vecino
que deja el menor costo C del camino (P_fallo si el camino no cumple los tipos de PE).
Termina cuando el camino cumple los tipos de PE y ningún vecino baja C, o cuando el PE actual no
tiene vecinos sin visitar.

Episodio: el camino terminado se convierte en el rectángulo que lo contiene (con columnas de
memoria al oeste y al este) y ese rectángulo se simula: su C es el costo real del episodio y la
recompensa es R = C_best_prev - C, con C_best_prev = el mejor C real hasta ahora (al principio,
el de la malla completa).

Ciclo de episodios (retroalimentación): se repiten episodios (semilla + n) hasta max_episodios o
hasta paciencia episodios seguidos sin que baje el mejor C. Si un episodio baja el mejor C, pasa
a ser el mejor; si no, se descarta. Un rectángulo ya simulado no se vuelve a simular. En el punto
de paro, con probabilidad epsilon el azar decide parar o seguir n PEs más (hasta max_extension), para
poder probar configuraciones más grandes que las que prefiere el estimado.

    python3 agente_greedy.py [--salida resultados/greedy]
    python3 agente_greedy.py --continuar --salida resultados/greedy   # reusa la simulación guardada

Resultados en --salida:

    evaluacion_malla.json  la evaluación de la malla: C, costos por operación y NOPs por PE
    pesos.json             el peso de cada PE y los datos con que se calculó
    camino.json            los PEs del camino en orden, con su tipo y peso, el mapa ('.' = fuera) y su
                           valuación con la función de costo (C, C_A, C_L, C_R), sin volver a simular
    historia.json          el camino paso a paso: PE agregado, quién lo eligió, el costo del prefijo
                           (C, C_A, C_L_norm, C_R) y la recompensa r_t = C(t-1) - C(t) en cada paso;
                           lo lee herramientas/gif_camino.py
    episodios.json         cada episodio: semilla, camino, decisiones, rectángulo, estado, C real,
                           C estimado, C_best_prev, recompensa, si mejoró o se repitió; el mejor y
                           por qué terminó el ciclo
    evaluaciones_episodios.json  las simulaciones de los rectángulos; --continuar las reusa
    camino.json / historia.json / resumen.json  son los del mejor episodio (o del último, si
                           ninguno fue correcto)
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
ARCHIVO_EPISODIOS = "episodios.json"
ARCHIVO_CACHE_EPISODIOS = "evaluaciones_episodios.json"


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _escribir_json(ruta, datos):
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def huella_entradas(ruta_costos=RUTA_COSTOS):
    """Huella (sha256) de lo que determina el resultado de una simulación además de la malla: el programa
    (``operaciones.c``), las ecuaciones de costo (``costos.json``) y los normalizadores. Una simulación
    guardada con otra huella no se reusa."""
    import hashlib
    from dijkstra.compilar import FUENTE
    h = hashlib.sha256()
    for ruta in (FUENTE, Path(ruta_costos), Path(RUTA_COSTOS).parent / "normalizacion.json"):
        h.update(Path(ruta).read_bytes())
    return h.hexdigest()


def leer_evaluacion(ruta, configuracion, huella=None):
    """La evaluación guardada de la malla, si es de esta misma configuración (y, con ``huella``, de las
    mismas entradas: programa, costos y normalizadores); si no, ``None``."""
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if datos.get("configuracion") != configuracion:
        return None
    if huella is not None and datos.get("huella") != huella:
        return None
    e = datos["evaluacion"]
    e["operaciones"] = {op: EvaluacionOperacion(**o) for op, o in e["operaciones"].items()}
    return Evaluacion(**e)


def _numero(valor):
    return "-" if valor is None else f"{valor:.6f}"


def _hora():
    return datetime.now().strftime("%H:%M:%S")


def _duracion(segundos):
    segundos = int(round(segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"


class MedidorConProgreso:
    """Envuelve ``medir`` para escribir en la terminal cada simulación: al empezar (con su tamaño), un
    latido cada ``latido`` segundos mientras corre (0 = sin latido) y al terminar (estado, C y tiempo)."""

    def __init__(self, medir, latido=60, escribir=print):
        self.medir, self.latido, self.escribir = medir, latido, escribir
        self.simulaciones = 0
        self.registro = []  # cada simulación: configuración, estado, C y segundos (para el reporte)

    def __call__(self, conf):
        import threading
        self.simulaciones += 1
        nombre = f"{conf['filas']}x{conf['columnas']}"
        self.escribir(f"[{_hora()}] simulación {self.simulaciones}: {nombre} ...", flush=True)
        t0 = time.monotonic()
        parar = threading.Event()

        def latir():
            while not parar.wait(self.latido):
                self.escribir(f"[{_hora()}]     ... {nombre} sigue simulando ({_duracion(time.monotonic() - t0)})",
                              flush=True)

        hilo = threading.Thread(target=latir, daemon=True) if self.latido and self.latido > 0 else None
        if hilo:
            hilo.start()
        try:
            e = self.medir(conf)
        finally:
            parar.set()
            if hilo:
                hilo.join()
        m = e.metricas or {}
        self.registro.append({"configuracion": nombre, "pes": conf["filas"] * conf["columnas"], "estado": e.estado,
                              "C": e.costo, "segundos": round(time.monotonic() - t0, 2)})
        self.escribir(f"[{_hora()}] simulación {self.simulaciones}: {nombre}: {e.estado}, C = {_numero(e.costo)}, "
                      f"C_A = {m.get('C_A')}, C_L_norm = {_numero(m.get('C_L_norm'))} "
                      f"en {_duracion(time.monotonic() - t0)}", flush=True)
        return e


def informes_finales(salida):
    """Al terminar la corrida: la tabla completa de resultados de todas las configuraciones evaluadas
    (``configuraciones.csv``, ``configuraciones_operaciones.csv``, ``configuraciones.json``) y las
    gráficas (frente de Pareto, curva de costos, curva de episodios y GIF). Las gráficas necesitan
    matplotlib, numpy y Pillow; si faltan, se avisa y la tabla se escribe igual."""
    from dijkstra import tabla
    evaluaciones, episodios = tabla.de_corrida(salida)
    registros = tabla.escribir(salida, tabla.construir(evaluaciones, episodios))
    eficientes = [r["nombre"] for r in registros if r["eficiente"]]
    print(f"Tabla de resultados: {len(registros)} configuraciones ({len(eficientes)} eficientes: {', '.join(eficientes)}) "
          f"en configuraciones.csv, configuraciones_operaciones.csv y configuraciones.json", flush=True)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "herramientas"))
        import gif_camino
        import pareto
    except ImportError as error:
        print(f"Gráficas no generadas ({error}). Instale matplotlib, numpy y Pillow (con uv: uv pip install numpy "
              f"matplotlib pillow) y corra: python3 herramientas/gif_camino.py {salida}; python3 "
              f"herramientas/pareto.py {salida} --exploracion", flush=True)
        return registros, evaluaciones
    correctas = sum(r["estado"] == "correcta" for r in registros)
    pareto.grafica(registros, Path(salida) / "frente_pareto.png",
                   f"Frente de Pareto de la exploración ({correctas} configuraciones evaluadas)")
    print(f"Gráficas: {Path(salida) / 'frente_pareto.png'}", flush=True)
    gif_camino.main([str(salida)])
    return registros, evaluaciones


def generar_reporte(salida, agente, medidor, fases, inicio_reloj, monitor):
    """``reporte_corrida.md`` y ``reporte_corrida.json``: tiempos, episodios, decisiones, Dijkstra,
    configuraciones y recursos (``dijkstra/reporte_corrida.py``)."""
    from dijkstra import reporte_corrida, tabla
    evaluaciones, episodios = tabla.de_corrida(salida)
    registros = tabla.construir(evaluaciones, episodios)
    fin = reporte_corrida.ahora()
    duracion = (fin - inicio_reloj).total_seconds()
    rep = reporte_corrida.construir(vars(agente.parametros), fases, inicio_reloj, fin, agente.historial,
                                    agente.motivo_fin, agente.c_best, medidor.registro, registros,
                                    evaluaciones, reporte_corrida.uso_cpu(duracion), monitor.detener())
    reporte_corrida.escribir(salida, rep)
    mon = rep["recursos"]["monitor"]
    print(f"Reporte: {Path(salida) / 'reporte_corrida.md'} (duración {reporte_corrida._duracion(duracion)}, "
          f"{rep['configuraciones']['evaluadas']} configuraciones, {rep['decisiones']['total']} decisiones, "
          f"CPU de los procesos de la corrida {reporte_corrida._duracion(mon.get('cpu_arbol_s'))}, "
          f"memoria pico {mon.get('memoria_arbol_pico_mb')} MB)", flush=True)
    return rep


def leer_cache_episodios(ruta, huella=None):
    """Las simulaciones guardadas de los rectángulos: ``{clave de la configuración: Evaluacion}``; con
    ``huella``, vacío si se guardaron con otras entradas (programa, costos o normalizadores)."""
    if not ruta.exists():
        return {}
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if huella is not None and datos.get("huella") != huella:
        return {}
    cache = {}
    for item in datos["episodios"]:
        e = item["evaluacion"]
        e["operaciones"] = {op: EvaluacionOperacion(**o) for op, o in e["operaciones"].items()}
        cache[Agente._clave(item["configuracion"])] = Evaluacion(**e)
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", default=Path("resultados/greedy"), type=Path,
                        help="Carpeta de resultados (por defecto resultados/greedy)")
    parser.add_argument("--parametros", default=RUTA_PARAMETROS, help="Parámetros del agente (config/agente.json)")
    parser.add_argument("--costos", default=RUTA_COSTOS, help="Ecuaciones y pesos w1-w4 (config/costos.json)")
    parser.add_argument("--cgra", default=RUTA_CGRA_JSON, help="Esquema de la CGRA (cgra/src/cgra.json)")
    parser.add_argument("--procesos", type=int, help="Procesos para simular las operaciones en paralelo")
    parser.add_argument("--mapeador-procesos", type=int,
                        help="Procesos que prueban los candidatos del mapeador (1 = secuencial; "
                             "mismo resultado; se limita al mayor número de candidatos de una operación)")
    parser.add_argument("--forzar-compilacion", action="store_true", help="Repetir los skills 02-03")
    parser.add_argument("--latido", type=float, default=60,
                        help="Segundos entre avisos mientras una simulación corre (0 = sin avisos)")
    parser.add_argument("--continuar", action="store_true",
                        help="Reusar la simulación de la malla guardada en --salida (si es la misma malla)")
    parser.add_argument("--epsilon", type=float, help="Cambia el epsilon de config/agente.json (0 a 1)")
    parser.add_argument("--paciencia", type=int, help="Cambia la paciencia de config/agente.json")
    parser.add_argument("--max-episodios", type=int, help="Cambia max_episodios de config/agente.json")
    parser.add_argument("--informes", action="store_true",
                        help="Además de convergencia.csv/.json, escribir las tablas, gráficas (Pareto, curvas, GIF) "
                             "y reporte_corrida.md (modo bloques)")
    args = parser.parse_args(argv)
    from dijkstra import reporte_corrida
    inicio_reloj = reporte_corrida.ahora()
    monitor = reporte_corrida.MonitorRecursos().iniciar()
    fases = {}
    t_fase = time.monotonic()

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
    if args.mapeador_procesos is not None:
        parametros.mapeador_procesos = args.mapeador_procesos
    if args.epsilon is not None:
        if not 0 <= args.epsilon <= 1:
            parser.exit(1, "--epsilon debe estar entre 0 y 1\n")
        parametros.epsilon = args.epsilon
    if args.paciencia is not None:
        parametros.paciencia = args.paciencia
    if args.max_episodios is not None:
        parametros.max_episodios = args.max_episodios

    operaciones = operaciones_del_c()
    try:
        compiladas = preparar(operaciones, forzar=args.forzar_compilacion)  # skills 02-03
        fases["compilación (skills 01-03)"] = time.monotonic() - t_fase
    except ErrorCompilacion as error:
        parser.exit(1, f"{error}\n")
    print(f"Skills 01-03: {', '.join(operaciones)}"
          f" ({'compiladas: ' + ', '.join(compiladas) if compiladas else 'IR, memoria y grafos al día'})")

    args.salida.mkdir(parents=True, exist_ok=True)
    inicio = time.monotonic()
    t_fase = time.monotonic()
    medidor = MedidorConProgreso(lambda c: medir_en_paralelo(c, operaciones, CARPETA_PROGRAMAS, args.costos,
                                                             parametros.t_max_simulacion_s, parametros.procesos,
                                                             parametros.mapeador_procesos), args.latido)
    agente = Agente(medidor, base, parametros, costos)
    malla = agente.malla()
    configuracion = agente.configuracion(malla)
    ruta_evaluacion = args.salida / ARCHIVO_EVALUACION
    huella = huella_entradas(args.costos)
    evaluacion = leer_evaluacion(ruta_evaluacion, configuracion, huella) if args.continuar else None
    if args.continuar and evaluacion is None and ruta_evaluacion.exists():
        print(f"No se reusa {ruta_evaluacion}: es de otra malla o se guardó con otro operaciones.c, costos.json "
              f"o normalizacion.json (huella distinta)", flush=True)
    if evaluacion is None:
        print(f"Simulando la malla {len(malla)}x{len(malla[0])} (Dijkstra + simulador, una vez; "
              f"en 32x32 tarda horas)...", flush=True)
        t_referencia = time.monotonic()
        evaluacion = agente.medir(configuracion)
        evaluacion.tiempo_s = round(time.monotonic() - t_referencia, 3)
        _escribir_json(ruta_evaluacion, {"configuracion": configuracion, "evaluacion": evaluacion.como_dict(),
                                         "huella": huella, "marca_tiempo": _ahora()})
    else:
        print(f"Malla {len(malla)}x{len(malla[0])}: se reusa la simulación de {ruta_evaluacion}")
    costo = "-" if evaluacion.costo is None else f"{evaluacion.costo:.6f}"
    print(f"Malla {len(malla)}x{len(malla[0])}: {evaluacion.estado}, C = {costo}")

    # Ciclo de episodios: cada uno recorre un camino, simula su rectángulo (salvo que ya esté
    # simulado) y su C real vuelve al agente (C_best, paciencia).
    ruta_cache = args.salida / ARCHIVO_CACHE_EPISODIOS
    if args.continuar:
        agente.cache_episodios = leer_cache_episodios(ruta_cache, huella)
        if not agente.cache_episodios and ruta_cache.exists():
            print(f"No se reusa {ruta_cache}: se guardó con otro operaciones.c, costos.json o normalizacion.json",
                  flush=True)
        if agente.cache_episodios:
            print(f"Se reusan {len(agente.cache_episodios)} simulaciones de rectángulos de {ruta_cache}")

    def guardar_cache(*_):
        _escribir_json(ruta_cache, {"episodios": [{"configuracion": json.loads(clave), "evaluacion": e.como_dict()}
                                                  for clave, e in agente.cache_episodios.items()],
                                    "huella": huella, "marca_tiempo": _ahora()})

    agente.al_simular = guardar_cache  # cada simulación nueva queda guardada (--continuar la reusa)
    print(f"Modo de exploración: {parametros.modo}"
          + (f" (bloques {', '.join(str(b) for b in parametros.bloques)})" if parametros.modo == "bloques" else ""),
          flush=True)

    def al_episodio(ep):
        guardar_cache()
        if ep.get("modo") == "bloques":
            pasos = [t for t in ep["trayectoria"] if t.get("estado") is not None]
            recorrido = " → ".join(f"{t['filas']}x{t['columnas']}" for t in pasos if t["aceptada"] and t.get("fase") != "tipos")
            print(f"Episodio {ep['n']} (semilla {ep['semilla']}): descenso {recorrido} "
                  f"({ep['simuladas']} mallas probadas, {ep['nuevas']} simuladas ahora, {ep['aceptadas']} aceptadas; "
                  f"fin: {ep['parada']}"
                  + (f"; fase de tipos: {ep['cambios_tipo']} cambios E/V aceptados, fin: {ep['parada_tipos']}"
                     if ep.get("parada_tipos") else "") + f"), {ep['estado']}, C real = {_numero(ep['C'])}, "
                  f"R = {_numero(ep['recompensa'])}{' → nuevo mejor' if ep['mejora'] else ''}", flush=True)
            return
        marca = "repetido, " if ep["repetido"] else ""
        if ep["extension"]:
            marca = f"siguió +{ep['extension']} PEs tras el punto de paro, " + marca
        print(f"Episodio {ep['n']} (semilla {ep['semilla']}): {len(ep['camino'])} PEs, rectángulo "
              f"{ep['filas']}x{ep['columnas']} ({' / '.join(ep['malla'])}), {marca}{ep['estado']}, "
              f"C real = {_numero(ep['C'])} (estimado {_numero(ep['C_estimado'])}), "
              f"R = {_numero(ep['recompensa'])}{' → nuevo mejor' if ep['mejora'] else ''}", flush=True)

    try:
        fases["malla de referencia (simulación o reuso)"] = time.monotonic() - t_fase
        t_fase = time.monotonic()
        agente.correr(evaluacion, al_episodio)
        fases["exploración (episodios)"] = time.monotonic() - t_fase
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    episodio = agente.episodio
    _escribir_json(args.salida / ARCHIVO_EPISODIOS, {
        "modo": parametros.modo, "malla_referencia": [" ".join(f) for f in malla],
        "max_episodios": parametros.max_episodios, "paciencia": parametros.paciencia,
        "epsilon": parametros.epsilon,
        "C_malla": evaluacion.costo, "C_best": agente.c_best, "motivo_fin": agente.motivo_fin,
        "mejor": None if agente.mejor is None else agente.mejor["n"],
        "episodios": agente.historial,
    })
    resultado = agente.resultado()
    _escribir_json(args.salida / "pesos.json", {
        "formula": "peso = w_tipo · (1 + NOPs/ciclos); w_tipo de pesos_area (w1-w4) en costos.json; "
                   "NOPs y ciclos del reporte de ciclos, sumados sobre las operaciones",
        "pesos": [asdict(p) for p in agente.pesos.values()],
    })
    _escribir_json(args.salida / "camino.json", resultado)
    if parametros.modo == "bloques":
        _escribir_json(args.salida / "resumen.json", {
            "id": "dijkstra-greedy", "actualizado": _ahora(), "modo": "bloques",
            "malla": f"{len(malla)}x{len(malla[0])}", "estado_malla": evaluacion.estado, "C_malla": evaluacion.costo,
            "bloques": list(parametros.bloques), "epsilon": parametros.epsilon, "semilla": resultado["semilla"],
            "episodios": len(agente.historial), "motivo_fin": agente.motivo_fin,
            "episodio_mejor": None if agente.mejor is None else agente.mejor["n"], "C_best": agente.c_best,
            "malla_episodio": episodio["malla"], "tamano_episodio": f"{episodio['filas']}x{episodio['columnas']}",
            "estado_episodio": episodio["estado"], "C_episodio": episodio["C"],
            "recompensa_episodio": episodio["recompensa"],
            "configuraciones_simuladas": len(agente.cache_episodios), "simulaciones_nuevas": medidor.simulaciones,
            "exploraciones": sum(ep["decisiones"].count("exploracion") for ep in agente.historial),
            "parametros_agente": vars(parametros), "tiempo_total_s": round(time.monotonic() - inicio, 1),
        })
        print(f"Ciclo: {len(agente.historial)} episodios (fin por {agente.motivo_fin}), "
              f"{len(agente.cache_episodios)} configuraciones evaluadas ({medidor.simulaciones} simuladas en esta "
              f"corrida); " + (f"mejor: episodio {agente.mejor['n']}, malla {episodio['filas']}x{episodio['columnas']}, "
                               f"C real = {_numero(agente.c_best)}" if agente.mejor else
                               f"ninguna malla mejoró el C de la referencia ({costo})")
              + f"; malla de referencia: C = {costo}")
        sys.path.insert(0, str(Path(__file__).resolve().parent / "herramientas"))
        import convergencia
        datos = convergencia.escribir(args.salida)
        print(f"Convergencia: C_ref = {_numero(datos['C_ref'])}, C_best = {_numero(datos['C_best'])}, "
              f"convergió en la evaluación {(datos['convergencia'] or {}).get('evaluacion')} de "
              f"{datos['evaluaciones']} ({_duracion((datos['convergencia'] or {}).get('tiempo_s') or 0)})", flush=True)
        if args.informes:  # las tablas, gráficas y el reporte de antes, solo si se piden
            t_fase = time.monotonic()
            informes_finales(args.salida)
            fases["informes (tablas y gráficas)"] = time.monotonic() - t_fase
            generar_reporte(args.salida, agente, medidor, fases, inicio_reloj, monitor)
        else:
            monitor.detener()
        print(f"Resultados en {args.salida}: convergencia.csv, convergencia.json (y, para --continuar, episodios.json, "
              f"evaluaciones_episodios.json, evaluacion_malla.json, camino.json, resumen.json)")
        return 0
    historia = agente.historia()
    _escribir_json(args.salida / "historia.json", historia)
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
        "parada": resultado["parada"],
        "episodios": len(agente.historial),
        "motivo_fin": agente.motivo_fin,
        "episodio_mejor": None if agente.mejor is None else agente.mejor["n"],
        "C_best": agente.c_best,
        "malla_episodio": episodio["malla"],
        "estado_episodio": episodio["estado"],
        "C_episodio": episodio["C"],
        "recompensa_episodio": episodio["recompensa"],
        "parametros_agente": vars(parametros),
        "tiempo_total_s": round(time.monotonic() - inicio, 1),
    })
    print(f"Camino epsilon-greedy (epsilon = {resultado['epsilon']}, semilla = {resultado['semilla']}) desde "
          f"{resultado['inicio']}: {resultado['pes']} PEs {resultado['por_tipo']}, "
          f"{resultado['exploraciones']} exploraciones, peso total {resultado['peso_total']:.6g}; "
          f"parada: {resultado['parada']}")
    v = resultado["valuacion"]
    print(f"Valuación del camino: C = {v['C']:.6f} (C_A = {v['C_A']:g}, C_L_norm = {v['C_L_norm']:.6g}, "
          f"C_R = {v['C_R']:.3g}); malla completa: C = {costo}")
    print(f"Ciclo: {len(agente.historial)} episodios (fin por {agente.motivo_fin}); "
          + (f"mejor: episodio {agente.mejor['n']}, rectángulo {' / '.join(agente.mejor['malla'])}, "
             f"C real = {_numero(agente.c_best)}" if agente.mejor else
             f"ningún episodio mejoró el C de la malla ({costo}); se guarda el último")
          + f"; malla completa: C = {costo}")
    mejor = historia["mejor"]
    if mejor is not None:
        print(f"Menor C entre los prefijos que cumplen los tipos de PE: {mejor['C']:.6f} en el paso "
              f"{mejor['paso']} ({mejor['pes']} PEs, hasta {mejor['pe']})")
    t_fase = time.monotonic()
    informes_finales(args.salida)
    fases["informes (tablas y gráficas)"] = time.monotonic() - t_fase
    generar_reporte(args.salida, agente, medidor, fases, inicio_reloj, monitor)
    print(f"Resultados en {args.salida}: reporte_corrida.md, episodios.json, camino.json, historia.json, pesos.json, "
          f"evaluacion_malla.json, evaluaciones_episodios.json, resumen.json (curva y GIF: python3 herramientas/gif_camino.py {args.salida})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
