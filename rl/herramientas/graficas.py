"""Gráficas de la comparación: las que pidió el profesor, a partir de los CSV de la comparación.

    uv run python -m rl graficas resultados/corridas/s1-comparacion-comparacion [OTRA_COMPARACION ...] --salida DIR

Lee ``curvas_epsilon.csv`` y ``comparacion_epsilon.csv`` de una o más carpetas de comparación (las del Sistema 1 y,
si se pasan, las del Sistema 2: las columnas son las mismas) y dibuja, con un panel por sistema:

    convergencia_por_evaluacion.png  el mejor C según el número de evaluaciones, una línea por epsilon
                                     (paciencia más alta): cómo se optimiza la función de costo
    convergencia_por_tiempo.png      lo mismo según el tiempo acumulado: qué tan rápido converge
    tiempo_convergencia.png          tiempo y evaluaciones hasta converger, por epsilon y paciencia
    mejor_costo.png                  el mejor C alcanzado por epsilon, una línea por paciencia

El archivo es el mismo en las dos ramas (en el Sistema 2: ``sistema2_greedy/herramientas/graficas.py``).
Necesita matplotlib.
"""

import csv
from pathlib import Path

NOMBRES = {"sistema1_llm": "Sistema 1 · LLM + Skills", "sistema2_greedy": "Sistema 2 · Greedy + Dijkstra"}
# rampa azul ordinal (paso 250 a 700 de la paleta de referencia): epsilon de menor a mayor
AZULES = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def _numero(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def leer(carpetas):
    """``(curvas, resumenes)`` de las carpetas de comparación, con los números convertidos."""
    curvas, resumenes = [], []
    for carpeta in carpetas:
        carpeta = Path(carpeta)
        with (carpeta / "curvas_epsilon.csv").open(encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                curvas.append({**fila, **{k: _numero(fila[k]) for k in ("epsilon", "paciencia", "n",
                                                                       "tiempo_acumulado_s", "C", "C_best")}})
        with (carpeta / "comparacion_epsilon.csv").open(encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                resumenes.append({**fila, **{k: _numero(fila[k]) for k in fila if k not in (
                    "sistema", "corrida", "motivo_fin", "mejor_malla")}})
    return curvas, resumenes


def _log_legible(ax):
    """Escala logarítmica con números normales (2, 1.5...) en lugar de 2×10⁰."""
    import matplotlib.ticker as ticker
    ax.set_yscale("log")
    formato = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(formato)
    ax.yaxis.set_minor_formatter(formato)
    ax.tick_params(axis="y", which="minor", labelsize=7, colors=TINTA_2)


def _unidad(segundos):
    """``(divisor, nombre)`` para mostrar tiempos: horas, minutos o segundos según el mayor."""
    maximo = max([s for s in segundos if s is not None] or [0])
    return (3600, "horas") if maximo >= 3600 else (60, "minutos") if maximo >= 60 else (1, "segundos")


def _color(epsilons, epsilon):
    orden = sorted(epsilons)
    i = orden.index(epsilon)
    return AZULES[round(i * (len(AZULES) - 1) / max(len(orden) - 1, 1))]


def _ejes(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _paneles(plt, sistemas, alto=4.6, compartir_y=True):
    fig, axes = plt.subplots(1, len(sistemas), figsize=(6.2 * len(sistemas), alto), sharey=compartir_y, squeeze=False)
    return fig, list(axes[0])


def _curvas_por(plt, curvas, x, etiqueta_x, ruta, escala_x=1.0):
    sistemas = sorted({c["sistema"] for c in curvas})
    fig, axes = _paneles(plt, sistemas)
    log = all(c["C_best"] is None or c["C_best"] > 0 for c in curvas)
    for ax, sistema in zip(axes, sistemas):
        propias = [c for c in curvas if c["sistema"] == sistema]
        p_max = max(c["paciencia"] for c in propias)
        propias = [c for c in propias if c["paciencia"] == p_max]
        epsilons = sorted({c["epsilon"] for c in propias})
        for e in epsilons:
            serie = sorted((c for c in propias if c["epsilon"] == e and c["C_best"] is not None), key=lambda c: c["n"])
            if not serie:
                continue
            xs = [c[x] / escala_x for c in serie]
            ax.step(xs, [c["C_best"] for c in serie], where="post", color=_color(epsilons, e), linewidth=2,
                    label=f"ε = {e:g}")
        ax.set_title(f"{NOMBRES.get(sistema, sistema)} (paciencia {p_max:g})", fontsize=10, color=TINTA)
        ax.set_xlabel(etiqueta_x, fontsize=9, color=TINTA_2)
        if log:
            _log_legible(ax)
        _ejes(ax)
    axes[0].set_ylabel("mejor costo C hasta ese punto", fontsize=9, color=TINTA_2)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return ruta


def _barras(plt, resumenes, ruta):
    sistemas = sorted({r["sistema"] for r in resumenes})
    fig, axes = plt.subplots(2, len(sistemas), figsize=(6.2 * len(sistemas), 7), squeeze=False, sharey="row")
    for j, sistema in enumerate(sistemas):
        propios = [r for r in resumenes if r["sistema"] == sistema]
        epsilons = sorted({r["epsilon"] for r in propios})
        paciencias = sorted({r["paciencia"] for r in propios})
        ancho = 0.8 / len(paciencias)
        divisor, unidad = _unidad([r["tiempo_convergencia_s"] for r in resumenes])
        for fila, (campo, escala, etiqueta) in enumerate((("tiempo_convergencia_s", divisor, f"{unidad} hasta converger"),
                                                          ("evaluacion_convergencia", 1, "evaluaciones hasta converger"))):
            ax = axes[fila][j]
            for k, p in enumerate(paciencias):
                valores = []
                for e in epsilons:
                    r = next((r for r in propios if r["epsilon"] == e and r["paciencia"] == p), None)
                    valores.append((r[campo] or 0) / escala if r and r[campo] is not None else 0)
                xs = [i + (k - (len(paciencias) - 1) / 2) * ancho for i in range(len(epsilons))]
                color = AZULES[round(k * (len(AZULES) - 1) / max(len(paciencias) - 1, 1))]
                ax.bar(xs, valores, width=ancho * 0.92, color=color, label=f"paciencia {p:g}", edgecolor="white",
                       linewidth=1)
            ax.set_xticks(range(len(epsilons)))
            ax.set_xticklabels([f"{e:g}" for e in epsilons])
            ax.set_xlabel("ε (probabilidad de explorar)", fontsize=9, color=TINTA_2)
            if j == 0:
                ax.set_ylabel(etiqueta, fontsize=9, color=TINTA_2)
            if fila == 0:
                ax.set_title(NOMBRES.get(sistema, sistema), fontsize=10, color=TINTA)
            _ejes(ax)
        axes[0][j].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return ruta


def _mejor(plt, resumenes, ruta):
    sistemas = sorted({r["sistema"] for r in resumenes})
    fig, axes = _paneles(plt, sistemas)
    for ax, sistema in zip(axes, sistemas):
        propios = [r for r in resumenes if r["sistema"] == sistema and r["C_best"] is not None]
        paciencias = sorted({r["paciencia"] for r in propios})
        for k, p in enumerate(paciencias):
            serie = sorted((r for r in propios if r["paciencia"] == p), key=lambda r: r["epsilon"])
            color = AZULES[round(k * (len(AZULES) - 1) / max(len(paciencias) - 1, 1))]
            ax.plot([r["epsilon"] for r in serie], [r["C_best"] for r in serie], color=color, linewidth=2,
                    marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=f"paciencia {p:g}")
        ax.set_title(NOMBRES.get(sistema, sistema), fontsize=10, color=TINTA)
        ax.set_xlabel("ε (probabilidad de explorar)", fontsize=9, color=TINTA_2)
        _ejes(ax)
    axes[0].set_ylabel("mejor costo C alcanzado", fontsize=9, color=TINTA_2)
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return ruta


def graficar(carpetas, salida):
    """Dibuja las cuatro gráficas en ``salida`` y devuelve sus rutas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    curvas, resumenes = leer(carpetas)
    divisor, unidad = _unidad([c["tiempo_acumulado_s"] for c in curvas])
    return [
        _curvas_por(plt, curvas, "n", "evaluaciones", salida / "convergencia_por_evaluacion.png"),
        _curvas_por(plt, curvas, "tiempo_acumulado_s", f"tiempo acumulado ({unidad})",
                    salida / "convergencia_por_tiempo.png", escala_x=divisor),
        _barras(plt, resumenes, salida / "tiempo_convergencia.png"),
        _mejor(plt, resumenes, salida / "mejor_costo.png"),
    ]
