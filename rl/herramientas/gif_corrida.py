"""Curva de costos y GIF de una corrida del Sistema 1 (como ``curva_costos.png`` y ``exploracion.gif`` del Sistema 2).

    uv run python -m rl informes --corrida resultados/corridas/<corrida>

    curva_costos.png  el C de cada configuración evaluada, en orden: las que mejoraron la mejor de su episodio
                      (círculo), las que no (x), quién las propuso (LLM o paso al azar) y el mejor C de la corrida
    exploracion.gif   un cuadro por evaluación: a la izquierda la malla propuesta (M, E y V con los colores del
                      Sistema 2, dibujada desde la esquina superior izquierda dentro del tamaño de la referencia) y a
                      la derecha la curva de costos hasta ese punto; el último cuadro es la mejor configuración

No evalúa nada: solo dibuja lo que guardó la corrida (``historial.jsonl``). Necesita matplotlib y Pillow.
"""

import io
from pathlib import Path

from .. import estados
from ..comun import leer_json, leer_jsonl
from .tabla import nombre_malla

COLORES = {"M": "#5b6c8f", "E": "#3fa796", "V": "#e0a030"}
COLOR_FONDO, COLOR_CURVA, COLOR_BEST, COLOR_LLM, COLOR_AZAR = "#ececec", "#2b4c7e", "#2a9d8f", "#2b4c7e", "#d94fa8"
MAX_CUADROS = 300


def puntos(carpeta):
    """Las evaluaciones en orden con lo que hace falta para dibujarlas: si mejoraron la mejor de su episodio (o, sin
    episodios, la mejor de la corrida) y el mejor C de la corrida hasta ahí."""
    carpeta = Path(carpeta)
    tolerancia = leer_json(carpeta / "corrida.json").get("tolerancia_empate_costo", 0.0)
    resultado, c_best, c_ref, mejor_ep = [], None, None, {}
    for n, e in enumerate(leer_jsonl(carpeta / "historial.jsonl")):
        correcta = e["estado_evaluacion"] == estados.CORRECTA and e.get("costo_total") is not None
        if e.get("es_referencia"):
            c_ref = e.get("costo_total")
        ep = (e.get("episodio") or {}).get("n")
        actual = mejor_ep.get(ep, c_ref) if ep is not None else c_best
        mejora = correcta and not e.get("es_referencia") and (actual is None or e["costo_total"] < actual - tolerancia)
        if mejora and ep is not None:
            mejor_ep[ep] = e["costo_total"]
        if correcta and (c_best is None or e["costo_total"] < c_best - tolerancia):
            c_best = e["costo_total"]
        resultado.append({"n": n, "id": e["id_evaluacion"], "malla": e.get("malla"), "C": e.get("costo_total"),
                          "estado": e["estado_evaluacion"], "correcta": correcta, "mejora": mejora, "C_best": c_best,
                          "decidio": "referencia" if e.get("es_referencia") else e.get("decidio"),
                          "episodio": ep, "paso": (e.get("episodio") or {}).get("paso"),
                          "repetida": bool(e.get("repetida"))})
    return resultado


def _curva(ax, pts, hasta=None):
    import matplotlib.ticker as ticker
    vista = pts if hasta is None else pts[:hasta + 1]
    correctos = [p for p in vista if p["correcta"]]
    inicios = sorted({p["n"] for p in vista if p["paso"] == 0})
    for x in inicios:
        ax.axvline(x - 0.5, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=1)
    ax.step([p["n"] for p in vista if p["C_best"] is not None], [p["C_best"] for p in vista if p["C_best"] is not None],
            where="post", color=COLOR_BEST, linewidth=2, alpha=0.85, label="mejor C de la corrida", zorder=2)
    for quien, color, etiqueta in (("llm", COLOR_LLM, "LLM"), ("exploracion", COLOR_AZAR, "paso al azar (ε)"),
                                   ("referencia", "#555555", "referencia 32x32")):
        si = [p for p in correctos if p["decidio"] == quien and (p["mejora"] or quien == "referencia")]
        no = [p for p in correctos if p["decidio"] == quien and not p["mejora"] and quien != "referencia"]
        if si:
            ax.scatter([p["n"] for p in si], [p["C"] for p in si], s=30, color=color, zorder=4,
                       label=f"{etiqueta}: mejora su episodio" if quien != "referencia" else etiqueta)
        if no:
            ax.scatter([p["n"] for p in no], [p["C"] for p in no], s=26, color=color, marker="x", zorder=3,
                       label=f"{etiqueta}: no mejora")
    if correctos and min(p["C"] for p in correctos) > 0:
        ax.set_yscale("log")
        formato = ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(formato)
        ax.yaxis.set_minor_formatter(formato)
        ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_xlim(-0.5, max(len(pts) - 0.5, 1))
    ax.set_xlabel("evaluación (en orden; las líneas punteadas separan los episodios)")
    ax.set_ylabel("costo C")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def curva_costos_png(carpeta, salida=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = puntos(carpeta)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _curva(ax, pts)
    episodios = len({p["episodio"] for p in pts if p["episodio"] is not None})
    ax.set_title(f"{len(pts) - 1} configuraciones evaluadas"
                 + (f" en {episodios} episodios" if episodios else ""))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    ruta = Path(salida or Path(carpeta) / "curva_costos.png")
    fig.savefig(ruta, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return ruta


def _malla(ax, malla, filas_max, cols_max):
    import numpy as np
    from matplotlib.colors import to_rgb
    imagen = np.ones((filas_max, cols_max, 3)) * np.array(to_rgb(COLOR_FONDO))
    for f, fila in enumerate(malla or []):
        for c, t in enumerate(fila.split()):
            imagen[f, c] = to_rgb(COLORES.get(t, "#999999"))
    ax.imshow(imagen, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORES[t], label=t) for t in COLORES], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)


def _texto(p):
    if p["decidio"] == "referencia":
        return f"Referencia {nombre_malla(p['malla'])}: C = {p['C']:.3f}"
    quien = "LLM" if p["decidio"] == "llm" else "paso al azar (ε)"
    lugar = f"Episodio {p['episodio']}, paso {p['paso']}: " if p["episodio"] is not None else ""
    costo = f"C = {p['C']:.3f}" if p["C"] is not None else p["estado"]
    resultado = "mejora su episodio" if p["mejora"] else ("no mejora" if p["correcta"] else p["estado"])
    return f"{lugar}{quien} propone {nombre_malla(p['malla']) if p['malla'] else '-'} · {costo} · {resultado}" \
           + (" · repetida" if p["repetida"] else "")


def gif_exploracion(carpeta, salida=None, ms=700, ms_final=3000):
    """Escribe el GIF; devuelve el número de cuadros (uno por evaluación, hasta ``MAX_CUADROS``, más el final)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    pts = puntos(carpeta)
    con_malla = [p for p in pts if p["malla"]]
    if not con_malla:
        return 0
    filas_max = max(len(p["malla"]) for p in con_malla)
    cols_max = max(len(p["malla"][0].split()) for p in con_malla)
    indices = list(range(len(pts)))
    if len(indices) > MAX_CUADROS:
        paso = len(indices) / MAX_CUADROS
        indices = sorted({int(i * paso) for i in range(MAX_CUADROS)} | {len(pts) - 1})
    mejor = min((p for p in pts if p["correcta"]), key=lambda p: (p["C"], p["n"]), default=None)
    cuadros = []

    def cuadro(i, titulo, malla):
        fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
        _malla(ax_m, malla, filas_max, cols_max)
        _curva(ax_c, pts, i)
        fig.suptitle(titulo, fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("P", palette=Image.ADAPTIVE)

    for i in indices:
        cuadros.append(cuadro(i, _texto(pts[i]), pts[i]["malla"]))
    if mejor is not None:
        cuadros.append(cuadro(len(pts) - 1, f"Mejor configuración: {nombre_malla(mejor['malla'])} · C = {mejor['C']:.3f}"
                                            f" (evaluación {mejor['n']})", mejor["malla"]))
    ruta = Path(salida or Path(carpeta) / "exploracion.gif")
    duraciones = [ms] * (len(cuadros) - 1) + [ms_final]
    cuadros[0].save(ruta, save_all=True, append_images=cuadros[1:], duration=duraciones, loop=0, optimize=False)
    return len(cuadros)
