"""Frente de Pareto área-latencia: evalúa una rejilla de configuraciones rectangulares y lo grafica.

    python3 herramientas/pareto.py resultados/<corrida> --listar        # qué se evaluaría (no simula)
    python3 herramientas/pareto.py resultados/<corrida>                 # evalúa la rejilla y grafica
    python3 herramientas/pareto.py resultados/<corrida> --filas 2-8 --interiores 1-8
    python3 herramientas/pareto.py resultados/<corrida> --solo-grafica  # redibuja desde pareto_rejilla.json
    python3 herramientas/pareto.py resultados/<corrida> --exploracion   # frente de lo que exploró el agente

Con ``--exploracion`` no hay rejilla ni simulaciones: el frente se arma con todas las configuraciones que
simuló la exploración de la corrida (``evaluaciones_episodios.json``, más la malla de referencia). Escribe
``frente_pareto.png`` y la tabla completa de resultados (``dijkstra/tabla.py``): ``configuraciones.csv``,
``configuraciones_operaciones.csv`` y ``configuraciones.json``. El modo rejilla escribe ``pareto_rejilla.json``,
``pareto_rejilla.csv`` y ``frente_pareto_rejilla.png``.

Cada configuración de la rejilla es un rectángulo de F filas con K columnas interiores, tomado de la
esquina de la malla de referencia (los tipos E/V de esas filas) y con una columna de memoria (M) al
oeste y otra al este: F x (K + 2) PEs. Se evalúa con el mismo evaluador del ciclo (Dijkstra +
simulador, ``medir_en_paralelo``) y con las ecuaciones de costos.json.

Como C_R es el mismo para todas las configuraciones correctas, los objetivos son el área (C_A) y la
latencia normalizada (C_L / C_L,ref). Una configuración es eficiente (óptima de Pareto) si ninguna otra
tiene a la vez menor o igual área y latencia, y menor en alguna de las dos.

Reusa lo ya simulado: la malla de referencia (``evaluacion_malla.json`` de la corrida), los rectángulos
de los episodios (``evaluaciones_episodios.json``) y lo que ya esté en ``pareto_rejilla.json`` (se guarda
después de cada configuración, así que una evaluación interrumpida se retoma donde quedó).

Escribe en la carpeta de la corrida (o en ``--salida``):

    pareto_rejilla.json  cada configuración con su estado, C, área, latencia, ciclos por operación, si es
                        eficiente y su origen (rejilla, referencia o episodio)
    pareto_rejilla.csv   la misma tabla, para la tesis
    frente_pareto_rejilla.png  soluciones eficientes y dominadas, la frontera de Pareto y el óptimo del agente
"""

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

RAIZ_DG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_DG))

ARCHIVO = "pareto_rejilla.json"


# -- rejilla ---------------------------------------------------------------------------------
def rango(texto):
    """``"2-5"`` -> [2, 3, 4, 5]; ``"3"`` -> [3]."""
    partes = [int(p) for p in str(texto).split("-")]
    if len(partes) == 1:
        partes *= 2
    if len(partes) != 2 or partes[0] < 1 or partes[1] < partes[0]:
        raise ValueError(f"rango inválido: {texto!r} (use por ejemplo 2-5)")
    return list(range(partes[0], partes[1] + 1))


def rectangulo(malla, letra_memoria, filas, interiores):
    """Las primeras ``filas`` filas de la malla, con sus primeras ``interiores`` columnas que no son
    solo de memoria, y una columna de memoria al oeste y otra al este."""
    columnas = [c for c in range(len(malla[0])) if any(malla[f][c] != letra_memoria for f in range(filas))]
    if filas > len(malla) or interiores > len(columnas):
        raise ValueError(f"la malla de referencia no tiene {filas} filas y {interiores} columnas interiores")
    columnas = columnas[:interiores]
    return tuple((letra_memoria,) + tuple(malla[f][c] for c in columnas) + (letra_memoria,) for f in range(filas))


def nombre(malla):
    return f"{len(malla)}x{len(malla[0])}"


# -- frente de Pareto ------------------------------------------------------------------------
from dijkstra.tabla import frente, marcar_frente  # noqa: E402


# -- evaluación ------------------------------------------------------------------------------
def registro(malla, evaluacion, origen):
    m = evaluacion.metricas or {}
    return {
        "nombre": nombre(malla), "filas": len(malla), "columnas": len(malla[0]),
        "malla": [" ".join(f) for f in malla], "origen": origen, "estado": evaluacion.estado,
        "motivo": evaluacion.motivo, "C": evaluacion.costo, "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
        "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"),
        "ciclos": {op: o.ciclos for op, o in evaluacion.operaciones.items()},
    }


def evaluar(mallas, configuracion, medir, cache=None, previos=None, al_registro=None):
    """Registros de cada malla. Usa ``previos`` (registros ya guardados, por malla) y ``cache``
    (``{clave de configuración: Evaluacion}``) antes de simular con ``medir``."""
    from dijkstra.agente import Agente
    cache, previos = cache or {}, previos or {}
    registros = []
    for malla in mallas:
        texto = tuple(" ".join(f) for f in malla)
        if texto in previos:
            registros.append(previos[texto])
            continue
        conf = configuracion(malla)
        evaluacion = cache.get(Agente._clave(conf))
        origen = "episodio" if evaluacion is not None else "rejilla"
        if evaluacion is None:
            evaluacion = medir(conf)
        r = registro(malla, evaluacion, origen)
        registros.append(r)
        if al_registro:
            al_registro(r)
    return registros


# -- gráfica ---------------------------------------------------------------------------------
def _curva_frontera(eficientes, log, extension=0.12):
    """La frontera de Pareto como curva suave, decreciente y convexa que sigue a los puntos eficientes
    (no una poligonal que los une uno por uno): se ajusta ``y = a + b·x^(-c)`` (b, c > 0) por mínimos
    cuadrados a los puntos eficientes y se dibuja un poco más allá de los extremos. Con menos de 3 puntos,
    sin numpy/scipy o si el ajuste falla, la poligonal de la envolvente convexa inferior de los puntos."""
    puntos = sorted({r["C_A"]: r["C_L_norm"] for r in eficientes}.items())
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    poligonal = _envolvente_inferior(puntos)
    if len(puntos) < 3:
        return [p[0] for p in poligonal], [p[1] for p in poligonal]
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return [p[0] for p in poligonal], [p[1] for p in poligonal]
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    escala = x[0]

    def modelo(v, a, b, c):
        return a + b * (v / escala) ** (-c)

    try:
        (a, b, c), _ = curve_fit(modelo, x, y, p0=(y.min(), max(y[0] - y.min(), 1e-3), 1.0),
                                 bounds=([-np.inf, 0, 0.05], [np.inf, np.inf, 10]), maxfev=20000)
    except (RuntimeError, ValueError):
        return [p[0] for p in poligonal], [p[1] for p in poligonal]
    if log:
        lo, hi = np.log10(x[0]), np.log10(x[-1])
        margen = (hi - lo) * extension
        muestra = 10 ** np.linspace(lo - margen * 0.4, hi + margen, 400)
    else:
        margen = (x[-1] - x[0]) * extension
        muestra = np.linspace(max(x[0] - margen * 0.4, 1e-9), x[-1] + margen, 400)
    return muestra, modelo(muestra, a, b, c)


def _envolvente_inferior(puntos):
    """La envolvente convexa inferior de ``puntos`` (ordenados por x): la frontera sin escalones."""
    casco = []
    for p in puntos:
        while len(casco) >= 2 and ((casco[-1][0] - casco[-2][0]) * (p[1] - casco[-2][1])
                                   - (casco[-1][1] - casco[-2][1]) * (p[0] - casco[-2][0])) <= 0:
            casco.pop()
        casco.append(p)
    return casco


def grafica(registros, ruta, titulo=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correctos = [r for r in registros if r["estado"] == "correcta" and r["C_A"] is not None]
    if not correctos:
        raise ValueError("no hay configuraciones correctas para graficar")
    eficientes = [r for r in correctos if r["eficiente"]]
    dominados = [r for r in correctos if not r["eficiente"]]
    eficientes.sort(key=lambda r: r["C_A"])
    mejor = min(correctos, key=lambda r: r["C"])
    fallidas = len(registros) - len(correctos)
    log = max(r["C_A"] for r in correctos) / min(r["C_A"] for r in correctos) > 20

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=130)
        fx, fy = _curva_frontera(eficientes, log)
        ax.plot(fx, fy, color="#222222", linestyle="--", linewidth=1.3, zorder=2, label="frontera de Pareto")
        ax.scatter([r["C_A"] for r in dominados], [r["C_L_norm"] for r in dominados], s=55, color="#4f7fd1",
                   zorder=3, label="solución dominada")
        ax.scatter([r["C_A"] for r in eficientes], [r["C_L_norm"] for r in eficientes], s=75, color="#4f7fd1",
                   edgecolor="#7cc243", linewidth=2.6, zorder=4, label="solución eficiente")
        ax.scatter([mejor["C_A"]], [mejor["C_L_norm"]], s=320, marker="*", facecolor="none", edgecolor="#7b3fa0",
                   linewidth=1.6, zorder=5, label=f"menor C ({mejor['nombre']}, C = {mejor['C']:.2f})")
        # etiquetas: las eficientes van numeradas (1, 2, ... por área) y su nombre en un recuadro, para que
        # no se encimen aunque haya varias mallas del mismo tamaño casi en el mismo lugar; de las dominadas
        # solo se nombran las grandes, que quedan separadas
        for i, r in enumerate(eficientes, start=1):
            ax.annotate(str(i), (r["C_A"], r["C_L_norm"]), textcoords="offset points", xytext=(7, 6),
                        fontsize=7.5, fontweight="bold", color="#3d7a1f")
        for r in dominados:
            if r.get("pes", r["filas"] * r["columnas"]) > 64:
                ax.annotate(r["nombre"], (r["C_A"], r["C_L_norm"]), textcoords="offset points", xytext=(6, 5),
                            fontsize=6.5, color="#777777")
        lista = "\n".join(f"{i}  {r['nombre']}   C = {r['C']:.3f}" for i, r in enumerate(eficientes, start=1))
        ax.text(0.27, 0.97, "Eficientes (tipos del interior,\nfila por fila, entre columnas M):\n" + lista,
                transform=ax.transAxes, va="top", ha="left", fontsize=7, family="DejaVu Sans Mono",
                color="#333333", bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"})
        if log:
            ax.set_xscale("log")
        ax.set_xlabel("Área $C_A$" + (" (escala logarítmica)" if log else ""))
        ax.set_ylabel("Latencia normalizada $C_L / C_{L,ref}$")
        ax.grid(True, color="#e6e6e6", linewidth=0.6, which="both")
        ax.set_title(titulo or f"Frente de Pareto área-latencia ({len(correctos)} configuraciones correctas"
                     + (f", {fallidas} fallidas" if fallidas else "") + ")")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.tight_layout()
        fig.savefig(ruta, metadata={"Software": None})
        plt.close(fig)


def escribir_csv(registros, ruta):
    operaciones = sorted({op for r in registros for op in r["ciclos"]})
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["configuracion", "filas", "columnas", "malla", "origen", "estado", "eficiente", "C", "C_A",
                    "C_A_norm", "C_L_norm", "C_R"] + [f"ciclos_{op}" for op in operaciones])
        for r in sorted(registros, key=lambda r: (r["C_A"] is None, r["C_A"] or 0, r["nombre"])):
            w.writerow([r["nombre"], r["filas"], r["columnas"], " / ".join(r["malla"]), r["origen"], r["estado"],
                        r.get("eficiente", False), r["C"], r["C_A"], r["C_A_norm"], r["C_L_norm"], r["C_R"]]
                       + [r["ciclos"].get(op) for op in operaciones])


def guardar(registros, salida, parametros, archivo=ARCHIVO):
    marcar_frente(registros)
    datos = {"objetivos": ["C_A", "C_L_norm"], "parametros": parametros,
             "configuraciones": sorted(registros, key=lambda r: (r["C_A"] is None, r["C_A"] or 0, r["nombre"]))}
    temporal = salida / (archivo + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporal.replace(salida / archivo)


def frente_de_exploracion(corrida, salida):
    """El frente de Pareto de las configuraciones que simuló la exploración de la corrida (sin simular):
    escribe la tabla completa de resultados (``dijkstra/tabla.py``) y ``frente_pareto.png``."""
    from dijkstra import tabla
    evaluaciones, episodios = tabla.de_corrida(corrida)
    if not evaluaciones:
        raise ValueError(f"{corrida} no tiene configuraciones evaluadas")
    registros = tabla.escribir(salida, tabla.construir(evaluaciones, episodios))
    correctas = sum(r["estado"] == "correcta" for r in registros)
    grafica(registros, Path(salida) / "frente_pareto.png",
            f"Frente de Pareto de la exploración ({correctas} configuraciones evaluadas)")
    return registros


# -- progreso -------------------------------------------------------------------------------
def _hora(momento=None):
    return (momento or datetime.now()).strftime("%H:%M:%S")


def _duracion(segundos):
    segundos = int(round(segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"


class Progreso:
    """Envuelve ``medir`` para escribir el avance en la terminal: una línea al empezar cada
    configuración, un latido cada ``latido`` segundos mientras se simula (0 = sin latido) y una línea al
    terminar con el resultado, el tiempo, cuántas faltan y la hora estimada de fin. La estimación usa
    el tiempo por PE de las configuraciones ya simuladas (las más grandes tardan más)."""

    def __init__(self, medir, pendientes, latido=60, escribir=print):
        self.medir = medir
        self.total = len(pendientes)
        self.pes_restantes = sum(len(m) * len(m[0]) for m in pendientes)
        self.latido = latido
        self.escribir = escribir
        self.hechas = 0
        self.segundos = 0.0
        self.pes_hechos = 0
        self.inicio = time.monotonic()

    def __call__(self, conf):
        nombre_conf = f"{conf['filas']}x{conf['columnas']}"
        pes = conf["filas"] * conf["columnas"]
        i = self.hechas + 1
        self.escribir(f"[{_hora()}] [{i}/{self.total}] simulando {nombre_conf} "
                      f"({' / '.join(conf['malla'])}) ...", flush=True)
        t0 = time.monotonic()
        parar = threading.Event()

        def latir():
            while not parar.wait(self.latido):
                self.escribir(f"[{_hora()}]     ... {nombre_conf} sigue simulando "
                              f"({_duracion(time.monotonic() - t0)})", flush=True)

        hilo = None
        if self.latido and self.latido > 0:
            hilo = threading.Thread(target=latir, daemon=True)
            hilo.start()
        try:
            evaluacion = self.medir(conf)
        finally:
            parar.set()
            if hilo is not None:
                hilo.join()
        dt = time.monotonic() - t0
        self.hechas += 1
        self.segundos += dt
        self.pes_hechos += pes
        self.pes_restantes -= pes
        faltan = self.total - self.hechas
        c = "-" if evaluacion.costo is None else f"{evaluacion.costo:.4f}"
        m = evaluacion.metricas or {}
        lat = m.get("C_L_norm")
        linea = (f"[{_hora()}] [{i}/{self.total}] {nombre_conf}: {evaluacion.estado}, C = {c}, "
                 f"C_A = {m.get('C_A')}, C_L_norm = {'-' if lat is None else f'{lat:.4f}'} "
                 f"en {_duracion(dt)}")
        if faltan:
            eta = self.segundos / self.pes_hechos * self.pes_restantes
            linea += (f" · faltan {faltan} · fin estimado ~{_hora(datetime.now() + timedelta(seconds=eta))} "
                      f"(en ~{_duracion(eta)})")
        else:
            linea += f" · simulaciones terminadas en {_duracion(time.monotonic() - self.inicio)}"
        self.escribir(linea, flush=True)
        return evaluacion


# -- comando ---------------------------------------------------------------------------------
def main(argv=None, medir=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corrida", type=Path, help="Carpeta de una corrida (con evaluacion_malla.json)")
    parser.add_argument("--filas", default="2-5", help="Filas F de los rectángulos (por defecto 2-5)")
    parser.add_argument("--interiores", default="1-4",
                        help="Columnas interiores K, sin contar las de memoria (por defecto 1-4)")
    parser.add_argument("--salida", type=Path, help="Carpeta de salida (por defecto la de la corrida)")
    parser.add_argument("--sin-referencia", action="store_true", help="No incluir la malla de referencia")
    parser.add_argument("--listar", action="store_true", help="Solo listar qué se evaluaría (no simula)")
    parser.add_argument("--solo-grafica", action="store_true", help="Redibujar desde pareto_rejilla.json (no simula)")
    parser.add_argument("--exploracion", action="store_true",
                        help="Frente de las configuraciones que simuló la exploración de la corrida (no simula)")
    parser.add_argument("--procesos", type=int, help="Procesos para las operaciones (por defecto, los de agente.json)")
    parser.add_argument("--mapeador-procesos", type=int, help="Procesos del mapeador (mismo resultado)")
    parser.add_argument("--latido", type=float, default=60,
                        help="Segundos entre avisos mientras una configuración se simula (0 = sin avisos)")
    args = parser.parse_args(argv)
    salida = args.salida or args.corrida
    salida.mkdir(parents=True, exist_ok=True)

    if args.exploracion:
        try:
            registros = frente_de_exploracion(args.corrida, salida)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        eficientes = [r["nombre"] for r in registros if r["eficiente"]]
        print(f"Frente de Pareto de la exploración: {len(registros)} configuraciones, {len(eficientes)} eficientes "
              f"({', '.join(eficientes)}). Resultados en {salida}: frente_pareto.png, configuraciones.csv, "
              f"configuraciones_operaciones.csv, configuraciones.json")
        return 0

    if args.solo_grafica:
        ruta = salida / ARCHIVO
        if not ruta.exists():
            parser.exit(1, f"{ruta} no existe\n")
        registros = marcar_frente(json.loads(ruta.read_text(encoding="utf-8"))["configuraciones"])
        grafica(registros, salida / "frente_pareto_rejilla.png")
        escribir_csv(registros, salida / "pareto_rejilla.csv")
        print(f"{salida / 'frente_pareto_rejilla.png'}")
        return 0

    from agente_greedy import leer_cache_episodios, leer_evaluacion
    from dijkstra import CARPETA_PROGRAMAS, leer_cgra
    from dijkstra import operaciones as operaciones_del_c
    from dijkstra.agente import Agente, Parametros, medir_en_paralelo
    from dijkstra.contrato import RUTA_COSTOS, Costos

    try:
        filas, interiores = rango(args.filas), rango(args.interiores)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    parametros = Parametros.cargar()
    agente = Agente(None, leer_cgra(), parametros, Costos.cargar())
    referencia = agente.malla()
    memoria = agente.letra_memoria()
    try:
        mallas = [rectangulo(referencia, memoria, f, k) for f in filas for k in interiores]
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    from dijkstra.evaluador import Evaluador
    validador = Evaluador([], CARPETA_PROGRAMAS, Costos.cargar())
    invalidas = {m: validador.motivos_invalida(agente.configuracion(m)) for m in mallas}
    validas = [m for m in mallas if not invalidas[m]]
    cache = leer_cache_episodios(args.corrida / "evaluaciones_episodios.json")
    previos = {}
    ruta = salida / ARCHIVO
    if ruta.exists():
        previos = {tuple(r["malla"]): r for r in json.loads(ruta.read_text(encoding="utf-8"))["configuraciones"]}
    pendientes = [m for m in validas if tuple(" ".join(f) for f in m) not in previos
                  and Agente._clave(agente.configuracion(m)) not in cache]

    print(f"Rejilla: filas {filas[0]}-{filas[-1]} x columnas interiores {interiores[0]}-{interiores[-1]}: "
          f"{len(mallas)} configuraciones, {len(validas)} válidas, {len(pendientes)} por simular")
    for m in mallas:
        texto = tuple(" ".join(f) for f in m)
        estado = ("inválida: " + "; ".join(invalidas[m]) if invalidas[m] else
                  "ya evaluada (pareto_rejilla.json)" if texto in previos else
                  "ya simulada (episodios)" if Agente._clave(agente.configuracion(m)) in cache else "por simular")
        print(f"  {nombre(m):>6}  {' / '.join(texto)}  [{estado}]")
    if args.listar:
        return 0

    evaluacion_malla = None if args.sin_referencia else leer_evaluacion(
        args.corrida / "evaluacion_malla.json", agente.configuracion(referencia))
    if not args.sin_referencia and evaluacion_malla is None:
        print("Aviso: la corrida no tiene la evaluación de la malla de referencia de agente.json; no se incluye")
    if medir is None:
        operaciones = operaciones_del_c()
        procesos = args.procesos if args.procesos is not None else parametros.procesos
        mapeador = args.mapeador_procesos if args.mapeador_procesos is not None else parametros.mapeador_procesos

        def medir(conf):
            return medir_en_paralelo(conf, operaciones, CARPETA_PROGRAMAS, RUTA_COSTOS,
                                     parametros.t_max_simulacion_s, procesos, mapeador)

    registros = []
    base = [registro(referencia, evaluacion_malla, "referencia")] if evaluacion_malla is not None else []
    inicio = time.monotonic()
    progreso = Progreso(medir, pendientes, args.latido)
    print(f"[{_hora()}] Inicio: {len(pendientes)} configuraciones por simular "
          f"({sum(len(m) * len(m[0]) for m in pendientes)} PEs en total); se guarda en {salida / ARCHIVO} "
          f"después de cada una", flush=True)

    def al_registro(r):
        registros.append(r)
        guardar(base + list(previos.values()) + registros, salida, vars(args) | {"salida": str(salida),
                                                                                    "corrida": str(args.corrida)})
        if r["origen"] == "episodio":
            print(f"[{_hora()}] {r['nombre']}: se reusa la simulación de los episodios", flush=True)

    todos = evaluar(validas, agente.configuracion, progreso, cache, previos, al_registro)
    finales = base + todos
    guardar(finales, salida, vars(args) | {"salida": str(salida), "corrida": str(args.corrida)})
    grafica(finales, salida / "frente_pareto_rejilla.png")
    escribir_csv(finales, salida / "pareto_rejilla.csv")
    eficientes = [r["nombre"] for r in finales if r["eficiente"]]
    print(f"Frente de Pareto ({len(eficientes)} eficientes de "
          f"{sum(r['estado'] == 'correcta' for r in finales)} correctas): {', '.join(eficientes)}")
    print(f"[{_hora()}] Listo en {_duracion(time.monotonic() - inicio)}. "
          f"Resultados en {salida}: pareto_rejilla.json, pareto_rejilla.csv, frente_pareto_rejilla.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
