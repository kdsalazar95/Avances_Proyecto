"""Curva de costo y GIF del camino de una corrida del agente Greedy, a partir de ``historia.json``.

    python3 herramientas/gif_camino.py resultados/<corrida>            # desde dijkstra_greedy/
    python3 herramientas/gif_camino.py resultados/<corrida> --ms 150 --max-cuadros 200

Escribe en la carpeta de la corrida (o en ``--salida``):

    curva_costo.png  C del prefijo del camino en cada paso, con los pasos del agente RL marcados y
                     el de menor C entre los que cumplen los tipos de PE (``mejor`` de la historia)
    camino.gif       un cuadro por paso: la malla con los PEs del camino resaltados y la curva hasta ese paso

Es determinística: la misma ``historia.json`` da los mismos archivos, byte por byte (sin fechas ni
versiones en los metadatos, sin azar). No simula ni llama al agente RL: solo dibuja lo que guardó la
corrida. Si la corrida es anterior a ``historia.json``, la reconstruye una vez con ``Agente.historia``
desde ``camino.json``, ``evaluacion_malla.json`` y ``resumen.json`` (sin simular) y la guarda.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

RAIZ_DG = Path(__file__).resolve().parents[1]
COLORES = {"M": "#5b6c8f", "E": "#3fa796", "V": "#e0a030"}
OTRO = "#999999"  # tipo sin color asignado
COLOR_CURVA, COLOR_RL, COLOR_MEJOR, COLOR_NO_CUMPLE = "#2b4c7e", "#d6453d", "#7b3fa0", "#b0b0b0"
ESTILO = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "svg.hashsalt": "gif_camino"}


# -- historia --------------------------------------------------------------------------------
def leer_historia(carpeta, reconstruir=True):
    ruta = Path(carpeta) / "historia.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    if not reconstruir:
        raise FileNotFoundError(f"{ruta} no existe")
    historia = reconstruir_historia(carpeta)
    ruta.write_text(json.dumps(historia, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{ruta}: reconstruida desde camino.json y evaluacion_malla.json")
    return historia


def reconstruir_historia(carpeta):
    """``historia.json`` de una corrida anterior a ella: el camino y sus decisiones de camino.json,
    los pesos y la valuación con la evaluación guardada de la malla (sin simular)."""
    carpeta = Path(carpeta)
    sys.path.insert(0, str(RAIZ_DG))
    from agente_greedy import leer_evaluacion
    from dijkstra import RUTA_CGRA_JSON, leer_cgra
    from dijkstra.agente import Agente, Parametros
    from dijkstra.contrato import Costos

    for nombre in ("camino.json", "evaluacion_malla.json", "resumen.json"):
        if not (carpeta / nombre).exists():
            raise FileNotFoundError(f"no hay historia.json ni {nombre} en {carpeta}")
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    datos = resumen["parametros_agente"]
    parametros = Parametros(**{k: datos[k] for k in Parametros.__dataclass_fields__ if k in datos})
    agente = Agente(None, leer_cgra(RUTA_CGRA_JSON), parametros, Costos.cargar())
    malla = agente.malla()
    evaluacion = leer_evaluacion(carpeta / "evaluacion_malla.json", agente.configuracion(malla))
    if evaluacion is None:
        raise ValueError(f"evaluacion_malla.json de {carpeta} no es de la malla de sus parámetros")
    agente.evaluacion = evaluacion
    agente.pesos = agente.calcular_pesos(evaluacion, malla)
    camino = json.loads((carpeta / "camino.json").read_text(encoding="utf-8"))["camino"]
    agente.camino = [p["pe"] for p in camino]
    agente.decisiones = [p["decision"] for p in camino]
    return agente.historia()


# -- dibujo ----------------------------------------------------------------------------------
def _rgb(color, claro=0.0):
    """Color hex a RGB en [0, 1]; ``claro`` lo mezcla con blanco."""
    c = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return c + (1 - c) * claro


def _malla_rgb(historia, visitados):
    tipos = [fila.split() for fila in historia["malla"]]
    imagen = np.ones((historia["filas"], historia["columnas"], 3))
    for f, fila in enumerate(tipos):
        for c, t in enumerate(fila):
            imagen[f, c] = _rgb(COLORES.get(t, OTRO), 0.0 if (f, c) in visitados else 0.8)
    return imagen


def _eje_curva(ax, historia):
    pasos = historia["pasos"]
    x = [p["pes"] for p in pasos]
    y = [p["C"] for p in pasos]
    margen = (max(y) - min(y)) * 0.08 or 0.05
    ax.set_xlim(0.5, max(x) + 0.5)
    ax.set_ylim(min(y) - margen, max(y) + margen)
    ax.set_xlabel("PEs del camino (k)")
    ax.set_ylabel("C del prefijo")
    ax.grid(True, color="#e4e4e4", linewidth=0.6)
    return x, y


def _dibujar_curva(ax, historia, hasta):
    """La curva hasta el paso ``hasta`` (incluido)."""
    pasos = historia["pasos"][:hasta + 1]
    x = [p["pes"] for p in pasos]
    y = [p["C"] for p in pasos]
    ax.plot(x, y, color=COLOR_CURVA, linewidth=1.4, zorder=2)
    no = [(p["pes"], p["C"]) for p in pasos if not p["cumple_tipos"]]
    if no:
        ax.scatter(*zip(*no), s=10, color=COLOR_NO_CUMPLE, zorder=3, label="no cumple tipos de PE")
    rl = [(p["pes"], p["C"]) for p in pasos if p["decision"] in ("rl", "rl_invalida", "exploracion")]
    if rl:
        ax.scatter(*zip(*rl), s=28, color=COLOR_RL, marker="D", zorder=4, label="paso del agente RL")
    mejor = historia.get("mejor")
    if mejor is not None and mejor["paso"] <= hasta:
        ax.scatter([mejor["pes"]], [mejor["C"]], s=120, color=COLOR_MEJOR, marker="*", zorder=5,
                   label=f"menor C válida: {mejor['C']:.4f} (k = {mejor['pes']})")
    ax.scatter([x[-1]], [y[-1]], s=30, facecolor="white", edgecolor=COLOR_CURVA, linewidth=1.4, zorder=6)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _titulo(historia):
    c = historia.get("C_malla")
    c = "-" if c is None else f"{c:.4f}"
    return (f"Malla {historia['filas']}x{historia['columnas']} (C = {c})  ·  "
            f"epsilon = {historia['epsilon']}, semilla = {historia['semilla']}")


def curva_png(historia, ruta):
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        _eje_curva(ax, historia)
        _dibujar_curva(ax, historia, len(historia["pasos"]) - 1)
        p = historia["pasos"][-1]
        ax.set_title(f"Costo del camino paso a paso: C final = {p['C']:.4f} con {p['pes']} PEs")
        fig.suptitle(_titulo(historia), fontsize=9, color="#555555")
        fig.tight_layout()
        fig.savefig(ruta, metadata={"Software": None})
        plt.close(fig)


def _cuadros(n, maximo):
    """Pasos que se dibujan: todos si caben en ``maximo``; si no, cada ``salto`` y siempre el último."""
    salto = max(1, math.ceil(n / maximo))
    indices = list(range(0, n, salto))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return indices


def gif(historia, ruta, ms=120, ms_final=2500, max_cuadros=300):
    pasos = historia["pasos"]
    cuadros = []
    with plt.rc_context(ESTILO):
        fig, (ax_malla, ax_curva) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=90,
                                                 gridspec_kw={"width_ratios": [1, 1.25]})
        for i in _cuadros(len(pasos), max_cuadros):
            ax_malla.clear()
            ax_curva.clear()
            visitados = {(p["fila"], p["columna"]) for p in pasos[:i + 1]}
            ax_malla.imshow(_malla_rgb(historia, visitados), interpolation="nearest")
            filas = [p["fila"] for p in pasos[:i + 1]]
            columnas = [p["columna"] for p in pasos[:i + 1]]
            ax_malla.plot(columnas, filas, color="#222222", linewidth=0.8)
            rl = [(p["columna"], p["fila"]) for p in pasos[:i + 1]
                  if p["decision"] in ("rl", "rl_invalida", "exploracion")]
            if rl:
                ax_malla.scatter(*zip(*rl), s=14, color=COLOR_RL, marker="D", zorder=3)
            actual = pasos[i]
            ax_malla.add_patch(Rectangle((actual["columna"] - 0.5, actual["fila"] - 0.5), 1, 1, fill=False,
                                         edgecolor="black", linewidth=1.6, zorder=4))
            ax_malla.set_xticks([])
            ax_malla.set_yticks([])
            ax_malla.set_title(f"Paso {actual['paso']}: {actual['pe']} ({actual['tipo']}, {actual['decision']})")
            ax_malla.legend(handles=[Patch(color=COLORES[t], label=t) for t in COLORES],
                            loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, frameon=False)
            _eje_curva(ax_curva, historia)
            _dibujar_curva(ax_curva, historia, i)
            ax_curva.set_title(f"k = {actual['pes']}:  C = {actual['C']:.4f}  (C_A = {actual['C_A']:g}, "
                               f"C_L_norm = {actual['C_L_norm']:.4f})")
            fig.suptitle(_titulo(historia), fontsize=9, color="#555555")
            fig.tight_layout()
            fig.canvas.draw()
            imagen = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            cuadros.append(imagen.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        plt.close(fig)
    duraciones = [ms] * (len(cuadros) - 1) + [ms_final]
    cuadros[0].save(ruta, save_all=True, append_images=cuadros[1:], duration=duraciones, loop=0,
                    optimize=False, disposal=1)
    return len(cuadros)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corrida", type=Path, help="Carpeta de resultados de la corrida (con historia.json)")
    parser.add_argument("--salida", type=Path, help="Carpeta donde escribir (por defecto la de la corrida)")
    parser.add_argument("--ms", type=int, default=120, help="Milisegundos por cuadro (por defecto 120)")
    parser.add_argument("--ms-final", type=int, default=2500, help="Milisegundos del último cuadro")
    parser.add_argument("--max-cuadros", type=int, default=300,
                        help="Cuadros como máximo; con caminos más largos se salta pasos (siempre incluye el último)")
    parser.add_argument("--sin-gif", action="store_true", help="Solo la curva (curva_costo.png)")
    parser.add_argument("--no-reconstruir", action="store_true",
                        help="Fallar si no hay historia.json en vez de reconstruirla")
    args = parser.parse_args(argv)

    try:
        historia = leer_historia(args.corrida, reconstruir=not args.no_reconstruir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo leer la historia: {error}\n")
    if not historia["pasos"]:
        parser.exit(1, "La historia no tiene pasos\n")
    salida = args.salida or args.corrida
    salida.mkdir(parents=True, exist_ok=True)
    curva_png(historia, salida / "curva_costo.png")
    print(f"{salida / 'curva_costo.png'}")
    if not args.sin_gif:
        n = gif(historia, salida / "camino.gif", args.ms, args.ms_final, args.max_cuadros)
        print(f"{salida / 'camino.gif'} ({n} cuadros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
