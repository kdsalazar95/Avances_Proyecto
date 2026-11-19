"""Curva de costo y GIF del camino de una corrida del agente Greedy, a partir de ``historia.json``.

    python3 herramientas/gif_camino.py resultados/<corrida>            # desde sistema2_greedy/
    python3 herramientas/gif_camino.py resultados/<corrida> --ms 150 --max-cuadros 200

Escribe en la carpeta de la corrida (o en ``--salida``):

    curva_costo.png  C del prefijo del camino en cada paso, con los pasos de exploración marcados y
                     el de menor C entre los que cumplen los tipos de PE (``mejor`` de la historia)
    camino.gif       un cuadro por paso: la malla con los PEs del camino resaltados y la curva hasta ese paso

Si la corrida tiene ``episodios.json`` (ciclo de episodios), además:

    curva_episodios.png  por episodio: C real (simulado) y C estimado, el mejor C hasta ese episodio
                         (C_best) y el desglose del C real en área (C_A_norm), latencia (C_L_norm) y error
    episodios.gif        un cuadro por episodio: el camino y su rectángulo sobre la esquina de la malla, y
                         la curva de C real hasta ese episodio

En el modo ``bloques`` (descenso por bloques desde la malla de referencia) no hay camino por PEs ni
``historia.json``; en su lugar:

    curva_costos.png       la curva de costos: el C real de cada malla probada, en orden (las aceptadas unidas
                           por episodio, las descartadas marcadas y el mejor C)
    exploracion.gif        la malla de PEs durante toda la reducción por bloques: por cada decisión, quién la
                           toma (Greedy o la exploración al azar), el bloque propuesto y el resultado (aceptada: la malla
                           se reduce; descartada: se marca en rojo), con la curva de costos al lado

Es determinística: la misma ``historia.json`` da los mismos archivos, byte por byte (sin fechas ni
versiones en los metadatos, sin azar). No simula: solo dibuja lo que guardó la
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
        ax.scatter(*zip(*rl), s=28, color=COLOR_RL, marker="D", zorder=4, label="paso de exploración")
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


# -- episodios -------------------------------------------------------------------------------
COLOR_ESTIMADO, COLOR_BEST, COLOR_AREA, COLOR_LAT, COLOR_ERR = "#9a9a9a", "#2a9d8f", "#4f7fd1", "#d08a2c", "#c0392b"


def leer_episodios(carpeta):
    ruta = Path(carpeta) / "episodios.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def _posicion(pe):
    """``PE0102`` -> (1, 2): la mitad de los dígitos es la fila y la otra la columna."""
    d = pe[2:]
    return int(d[:len(d) // 2]), int(d[len(d) // 2:])


def _eje_costo_episodios(ax_c, episodios, hasta, datos):
    """C real, C estimado y C_best de los episodios hasta ``hasta`` (incluido); escala de todos."""
    eps = episodios[:hasta + 1]
    n = [e["n"] for e in episodios]
    correctos = [e for e in eps if e["estado"] == "correcta" and e["C"] is not None]
    reales = [e["C"] for e in episodios if e["C"] is not None] + [e["C_estimado"] for e in episodios]
    alto, bajo = max(reales), min(reales)
    margen = (alto - bajo) * 0.12 or 0.1
    # C real, estimado y C_best
    ax_c.set_xlim(min(n) - 0.5, max(n) + 0.5)
    ax_c.set_ylim(max(0, bajo - margen), alto + margen)
    ax_c.set_xticks(n)
    ax_c.set_xlabel("episodio")
    ax_c.set_ylabel("costo C")
    ax_c.grid(True, color="#e4e4e4", linewidth=0.6)
    ax_c.plot([e["n"] for e in eps], [e["C_estimado"] for e in eps], color=COLOR_ESTIMADO, linestyle="--",
              marker="o", markersize=4, linewidth=1, label="C estimado (sin simular)")
    if correctos:
        ax_c.plot([e["n"] for e in correctos], [e["C"] for e in correctos], color=COLOR_CURVA, linewidth=1.4,
                  marker="o", markersize=5, label="C real (simulado)")
        best = [(e["n"], e["C_best"]) for e in eps if e.get("C_best") is not None and e["C_best"] <= alto + margen]
        if best:
            ax_c.step(*zip(*best), where="post", color=COLOR_BEST, linewidth=2, alpha=0.8, label="C_best")
        for e in correctos:
            if e.get("mejora"):
                ax_c.scatter([e["n"]], [e["C"]], s=140, marker="*", color=COLOR_MEJOR, zorder=5)
            if e.get("extension"):
                ax_c.annotate(f"SEGUIR {e['extension']}", (e["n"], e["C"]), textcoords="offset points",
                              xytext=(0, 9), ha="center", fontsize=7.5, color=COLOR_LAT)
            if e.get("repetido"):
                ax_c.annotate("repetido", (e["n"], e["C"]), textcoords="offset points", xytext=(0, -13),
                              ha="center", fontsize=7, color="#777777")
    ax_c.scatter([], [], s=140, marker="*", color=COLOR_MEJOR, label="nuevo mejor")
    ax_c.legend(loc="upper left", fontsize=7.5, frameon=False)
    c_malla = datos.get("C_malla")
    ax_c.set_title(f"C real por episodio (malla de referencia: C = {c_malla:.2f})" if c_malla is not None
                   else "C real por episodio")


def _eje_desglose(ax_t, episodios, hasta):
    """Barras apiladas del C real de cada episodio: área, latencia y error normalizados."""
    eps = episodios[:hasta + 1]
    n = [e["n"] for e in episodios]
    ax_t.set_xlim(min(n) - 0.6, max(n) + 0.6)
    ax_t.set_xticks(n)
    ax_t.set_xlabel("episodio")
    ax_t.set_ylabel("términos normalizados")
    ax_t.grid(True, axis="y", color="#e4e4e4", linewidth=0.6)
    tope = max((e.get("C_A_norm") or 0) + (e.get("C_L_norm") or 0) + (e.get("C_R") or 0) for e in episodios)
    ax_t.set_ylim(0, tope * 1.15 or 1)
    for e in eps:
        a, l, r = e.get("C_A_norm") or 0, e.get("C_L_norm") or 0, e.get("C_R") or 0
        ax_t.bar(e["n"], a, color=COLOR_AREA, width=0.6)
        ax_t.bar(e["n"], l, bottom=a, color=COLOR_LAT, width=0.6)
        if r:
            ax_t.bar(e["n"], r, bottom=a + l, color=COLOR_ERR, width=0.6)
        ax_t.text(e["n"], a + l + r, f"{e['filas']}x{e['columnas']}", ha="center", va="bottom", fontsize=7.5)
    ax_t.legend(handles=[Patch(color=COLOR_AREA, label="área C_A/C_A,ref"),
                         Patch(color=COLOR_LAT, label="latencia C_L/C_L,ref"),
                         Patch(color=COLOR_ERR, label="error C_R")], loc="upper left", fontsize=7.5, frameon=False)
    ax_t.set_title("Desglose del C real (rectángulo simulado)")


def _titulo_episodios(datos):
    return (f"{len(datos['episodios'])} episodios · fin por {datos.get('motivo_fin')} · "
            f"epsilon = {datos.get('epsilon')} · mejor C = {datos.get('C_best'):.4f}"
            if datos.get("C_best") is not None else f"{len(datos['episodios'])} episodios")


def curva_episodios_png(datos, ruta):
    episodios = datos["episodios"]
    with plt.rc_context(ESTILO):
        fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
        _eje_costo_episodios(ax_c, episodios, len(episodios) - 1, datos)
        _eje_desglose(ax_t, episodios, len(episodios) - 1)
        fig.suptitle(_titulo_episodios(datos), fontsize=9, color="#555555")
        fig.tight_layout()
        fig.savefig(ruta, metadata={"Software": None})
        plt.close(fig)


def gif_episodios(datos, historia, ruta, ms=1600, ms_final=4000):
    episodios = datos["episodios"]
    tipos = [fila.split() for fila in historia["malla"]]
    posiciones = [[_posicion(pe) for pe in e["camino"]] for e in episodios]
    f_max = min(len(tipos), max(f for ps in posiciones for f, _ in ps) + 3)
    c_max = min(len(tipos[0]), max(c for ps in posiciones for _, c in ps) + 3)
    cuadros = []
    with plt.rc_context(ESTILO):
        fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=90, gridspec_kw={"width_ratios": [0.8, 1.2]})
        for i, e in enumerate(episodios):
            ax_m.clear()
            ax_c.clear()
            ps = posiciones[i]
            imagen = np.ones((f_max, c_max, 3))
            for f in range(f_max):
                for c in range(c_max):
                    imagen[f, c] = _rgb(COLORES.get(tipos[f][c], OTRO), 0.0 if (f, c) in ps else 0.75)
            ax_m.imshow(imagen, interpolation="nearest")
            ax_m.plot([c for _, c in ps], [f for f, _ in ps], color="#222222", linewidth=1.2, marker="o", markersize=3)
            f0, f1 = min(f for f, _ in ps), max(f for f, _ in ps)
            c0, c1 = min(c for _, c in ps), max(c for _, c in ps)
            ax_m.add_patch(Rectangle((c0 - 0.5, f0 - 0.5), c1 - c0 + 1, f1 - f0 + 1, fill=False,
                                     edgecolor=COLOR_MEJOR, linewidth=2, linestyle="--"))
            ax_m.set_xticks(range(c_max))
            ax_m.set_yticks(range(f_max))
            ax_m.tick_params(labelsize=7)
            marca = " (repetido)" if e.get("repetido") else ""
            ax_m.set_title(f"Episodio {e['n']}: {len(ps)} PEs → {e['filas']}x{e['columnas']}{marca}\n"
                           f"{' / '.join(e['malla'])}", fontsize=8.5)
            ax_m.legend(handles=[Patch(color=COLORES[t], label=t) for t in COLORES],
                        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False)
            _eje_costo_episodios(ax_c, episodios, i, datos)
            c_txt = "-" if e["C"] is None else f"{e['C']:.4f}"
            r_txt = "-" if e["recompensa"] is None else f"{e['recompensa']:+.4f}"
            ax_c.set_title(f"Episodio {e['n']}: C real = {c_txt}, R = {r_txt}"
                           + ("  → nuevo mejor" if e.get("mejora") else ""))
            fig.suptitle(_titulo_episodios(datos), fontsize=9, color="#555555")
            fig.tight_layout()
            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            cuadros.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        plt.close(fig)
    duraciones = [ms] * (len(cuadros) - 1) + [ms_final]
    cuadros[0].save(ruta, save_all=True, append_images=cuadros[1:], duration=duraciones, loop=0,
                    optimize=False, disposal=1)
    return len(cuadros)


# -- modo bloques ----------------------------------------------------------------------------
COLOR_ACEPTADA, COLOR_DESCARTADA, COLOR_PROPUESTA, COLOR_QUITADO = "#2a9d8f", "#c0392b", "#d94fa8", "#e9e9e9"
QUIEN = {"explotacion": "Greedy", "exploracion": "Exploración al azar", "inicio": "inicio",
         # corridas híbridas anteriores a la separación de los sistemas (el agente consultaba un LLM)
         "rl": "Agente LLM (corrida híbrida)", "rl_invalida": "Greedy (respuesta del LLM inválida)"}


def pasos_exploracion(datos):
    """Las mallas probadas en orden, ``(episodio, paso)``, con el punto de partida (la malla de referencia)
    al comienzo de cada episodio para que cada descenso se vea completo; ``probadas`` no lo cuenta."""
    pasos = []
    for ep in datos["episodios"]:
        for t in ep.get("trayectoria", []):
            if t.get("estado") is None:
                continue
            pasos.append((ep["n"], t))
    return pasos


def nombre_malla(malla):
    """El nombre de la configuración con la posición de cada tipo de PE (``dijkstra/tabla.py``)."""
    if str(RAIZ_DG) not in sys.path:
        sys.path.insert(0, str(RAIZ_DG))
    from dijkstra.tabla import nombre_malla as nombre
    return nombre(malla)


def probadas(pasos):
    """Cuántas mallas se probaron (sin contar el punto de partida repetido de cada episodio)."""
    return sum(1 for _, t in pasos if t["accion"] != "inicio") + (1 if pasos else 0)


def etiqueta(t):
    """Etiqueta de un punto de la curva: el tamaño, o el cambio de tipo en la fase de tipos."""
    if t.get("fase") == "tipos":
        return f"{t['de']}→{t['a']}"
    return f"{t['filas']}x{t['columnas']}"


def mejor_paso(datos):
    """El paso (de cualquier episodio) con la configuración correcta de menor C real."""
    pasos = [t for _, t in pasos_exploracion(datos) if t["estado"] == "correcta" and t["C"] is not None]
    return min(pasos, key=lambda t: t["C"], default=None)


class MallaPEs:
    """Dibuja la malla de referencia como PEs (celdas por tipo) con la configuración activa de un
    rectángulo como un bloque continuo: sus columnas interiores y, a los lados, sus columnas de memoria."""

    def __init__(self, malla_referencia, tam=9):
        self.tipos = [fila.split() for fila in malla_referencia]
        self.filas, self.columnas = len(self.tipos), len(self.tipos[0])
        memoria = [c for c in range(self.columnas) if all(self.tipos[f][c] == "M" for f in range(self.filas))]
        self.interiores = [c for c in range(self.columnas) if c not in memoria]
        self.tam = tam

    def activos(self, rect, malla=None):
        """``{(fila, columna): tipo}`` de la configuración: las columnas interiores ``[k0, k1)`` y una
        columna M a cada lado, pegadas a ellas (como en la configuración que se simula). Con ``malla``
        (las filas de la configuración, ``"M E V M"``), los tipos del interior salen de ella (la fase de
        tipos cambia PEs E <-> V y la malla ya no coincide con la referencia)."""
        f0, f1, k0, k1 = rect
        columnas = [self.interiores[k] for k in range(k0, k1)]
        oeste, este = max(0, columnas[0] - 1), min(self.columnas - 1, columnas[-1] + 1)
        celdas = {}
        for f in range(f0, f1):
            celdas[(f, oeste)] = "M"
            celdas[(f, este)] = "M"
            propias = malla[f - f0].split()[1:-1] if malla else None
            for j, c in enumerate(columnas):
                celdas[(f, c)] = propias[j] if propias and len(propias) == len(columnas) else self.tipos[f][c]
        return celdas

    def imagen(self, activos, resaltados=(), color_resaltado=COLOR_PROPUESTA):
        t = self.tam
        img = np.ones((self.filas * t + 1, self.columnas * t + 1, 3))
        resaltados = set(resaltados)
        for f in range(self.filas):
            for c in range(self.columnas):
                if (f, c) in resaltados:
                    color = _rgb(color_resaltado)
                elif (f, c) in activos:
                    color = _rgb(COLORES.get(activos[(f, c)], OTRO))
                else:
                    color = _rgb(COLOR_QUITADO)
                img[f * t + 1:(f + 1) * t, c * t + 1:(c + 1) * t] = color
        return img

    def contorno(self, ax, celdas, color, estilo="-", ancho=2.0):
        if not celdas:
            return
        fs = [f for f, _ in celdas]
        cs = [c for _, c in celdas]
        t = self.tam
        ax.add_patch(Rectangle((min(cs) * t + 0.5, min(fs) * t + 0.5), (max(cs) - min(cs) + 1) * t,
                               (max(fs) - min(fs) + 1) * t, fill=False, edgecolor=color, linewidth=ancho,
                               linestyle=estilo, zorder=3))

    @staticmethod
    def ampliacion(ax, activos, resaltados=(), color_resaltado=COLOR_PROPUESTA):
        """La configuración activa ampliada, con la letra de cada PE (para mallas pequeñas)."""
        filas = sorted({f for f, _ in activos})
        columnas = sorted({c for _, c in activos})
        img = np.ones((len(filas), len(columnas), 3))
        for i, f in enumerate(filas):
            for j, c in enumerate(columnas):
                color = color_resaltado if (f, c) in set(resaltados) else COLORES.get(activos[(f, c)], OTRO)
                img[i, j] = _rgb(color)
        ax.imshow(img, interpolation="nearest")
        for i, f in enumerate(filas):
            for j, c in enumerate(columnas):
                ax.text(j, i, activos[(f, c)], ha="center", va="center", fontsize=8 if len(columnas) <= 12 else 6,
                        color="white", fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(columnas), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(filas), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(f"configuración {len(filas)}x{len(columnas)}", fontsize=8)


def _eje_costos(ax, puntos, actual=None):
    """C real de las mallas probadas: ``puntos`` = [(orden, episodio, paso)]."""
    ax.set_yscale("log")
    ax.set_xlabel("malla probada (en orden)")
    ax.set_ylabel("C real (escala logarítmica)")
    ax.grid(True, color="#e4e4e4", linewidth=0.6, which="both")
    if not puntos:
        return
    episodios = sorted({n for _, n, _ in puntos})
    for n in episodios:
        aceptadas = [(i, t["C"]) for i, m, t in puntos if m == n and t["aceptada"] and t["C"] is not None]
        if aceptadas:
            ax.plot(*zip(*aceptadas), color=COLOR_CURVA, linewidth=1.4, marker="o", markersize=5, zorder=3)
    descartadas = [(i, t["C"]) for i, _, t in puntos if not t["aceptada"] and t["C"] is not None]
    if descartadas:
        ax.scatter(*zip(*descartadas), marker="x", s=50, color=COLOR_DESCARTADA, zorder=4)
    anterior = None
    for i, n, t in puntos:
        if anterior is not None and n != anterior:
            ax.axvline(i - 0.5, color="#bbbbbb", linestyle=":", linewidth=1)
        anterior = n
        if t["C"] is not None and not (t.get("fase") == "tipos" and not t["aceptada"]):
            ax.annotate(etiqueta(t), (i, t["C"]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.5 if t.get("fase") == "tipos" else 7, color="#444444")
    if actual is not None and actual[2]["C"] is not None:
        ax.scatter([actual[0]], [actual[2]["C"]], s=170, facecolor="none", edgecolor="black", linewidth=1.6, zorder=5)
    tipos = [(i, t["C"]) for i, _, t in puntos if t.get("fase") == "tipos" and t["aceptada"] and t["C"] is not None]
    if tipos:
        ax.scatter(*zip(*tipos), marker="D", s=34, color=COLOR_MEJOR, zorder=4)
        ax.scatter([], [], marker="D", color=COLOR_MEJOR, label="cambio de tipo aceptado (E/V)")
    ax.plot([], [], color=COLOR_CURVA, marker="o", label="aceptada (descenso)")
    ax.scatter([], [], marker="x", color=COLOR_DESCARTADA, label="descartada")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def _limites_costos(ax, todos, n):
    ax.set_xlim(-0.6, max(n, 1) - 0.4)
    ax.set_ylim(min(todos) * 0.8, max(todos) * 1.3)


def curva_costos_png(datos, ruta):
    """La curva de costos de la exploración: el C real de cada malla probada, en orden."""
    pasos = pasos_exploracion(datos)
    puntos = [(i, n, t) for i, (n, t) in enumerate(pasos)]
    todos = [t["C"] for _, _, t in puntos if t["C"] is not None]
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
        _eje_costos(ax, puntos)
        _limites_costos(ax, todos, len(puntos))
        mejor = mejor_paso(datos)
        if mejor is not None:
            ax.axhline(mejor["C"], color=COLOR_MEJOR, linestyle="--", linewidth=1,
                       label=f"mejor: {nombre_malla(mejor['malla'])}, C = {mejor['C']:.3f}")
        # la leyenda debajo de la gráfica: arriba están las etiquetas de la malla de referencia
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8, frameon=False)
        distintas = len({tuple(t["malla"]) for _, t in pasos if t.get("malla")})
        ax.set_title(f"Curva de costos de la exploración: {probadas(pasos)} propuestas evaluadas "
                     f"({distintas} mallas distintas) en "
                     f"{len(datos['episodios'])} episodios")
        fig.suptitle(_titulo_episodios(datos), fontsize=9, color="#555555")
        fig.tight_layout()
        fig.savefig(ruta, metadata={"Software": None})
        plt.close(fig)


def anclar(rect):
    """El rectángulo equivalente anclado en la esquina superior izquierda (PE00). En la referencia uniforme
    (columnas interiores iguales en cada fila, filas alternas E/V) la configuración solo depende del tamaño y
    de si la primera fila es E o V (la paridad de la fila inicial), no de dónde está el rectángulo: así se
    dibuja siempre en el mismo lugar, aunque la corrida lo haya guardado en otra esquina."""
    f0, f1, k0, k1 = rect
    inicio = f0 % 2
    return [inicio, inicio + (f1 - f0), 0, k1 - k0]


def accion_vista(antes, despues):
    """El nombre de la acción según cómo se ve en el dibujo anclado (de dónde salen las filas o columnas)."""
    a, d = anclar(antes), anclar(despues)
    if d[1] - d[0] < a[1] - a[0]:
        return "FILAS INICIO" if d[0] != a[0] else "FILAS FIN"
    return "COLUMNAS FIN"


def cuadros_exploracion(datos):
    """La secuencia de cuadros del GIF: para cada decisión, la propuesta y su resultado."""
    cuadros, orden = [], 0
    for ep in datos["episodios"]:
        trayectoria = ep.get("trayectoria", [])
        if not trayectoria:
            continue
        inicio = trayectoria[0]
        rect, c_actual, actual = anclar(inicio["estado_rect"]), inicio["C"], None
        if orden == 0:
            cuadros.append({"tipo": "inicio", "ep": ep["n"], "rect": rect, "punto": (0, ep["n"], inicio),
                            "texto": f"Malla de referencia {inicio['filas']}x{inicio['columnas']}: "
                                     f"C real = {inicio['C']:.3f}. El agente empieza a reducirla por bloques."})
            orden = 1
        else:
            cuadros.append({"tipo": "inicio", "ep": ep["n"], "rect": rect, "punto": (orden, ep["n"], inicio),
                            "texto": f"Episodio {ep['n']}: el descenso vuelve a empezar en la malla de referencia"})
            orden += 1
        for t in trayectoria[1:]:
            if t["accion"] in ("BAJAR", "PARAR"):
                que = ("pasar al bloque siguiente" if t["accion"] == "BAJAR" else
                       "terminar la fase de tipos" if t.get("fase") == "tipos" else "terminar el descenso")
                cuadros.append({"tipo": "control", "ep": ep["n"], "rect": rect, "malla": actual,
                                "bloque": t["bloque"], "fase": t.get("fase"),
                                "texto": f"El agente LLM (corrida híbrida) decide {t['accion']}: {que}"})
                continue
            quien = QUIEN.get(t["quien"], t["quien"])
            if t.get("fase") == "tipos":
                propuesta = (f"{quien} propone cambiar el PE (fila {t['pe'][0]}, columna {t['pe'][1]}) de {t['de']} "
                             f"a {t['a']}: C estimado {t['C_estimado']:.3f}")
            else:
                propuesta = (f"{quien} propone {accion_vista(rect, t['estado_rect'])} (bloque {t['bloque']}): "
                             f"→ {t['filas']}x{t['columnas']}, C estimado {t['C_estimado']:.3f}")
            cuadros.append({"tipo": "propuesta", "ep": ep["n"], "rect": rect, "malla": actual,
                            "candidato": anclar(t["estado_rect"]), "candidato_malla": t.get("malla"),
                            "bloque": t["bloque"], "fase": t.get("fase"), "texto": propuesta})
            punto = (orden, ep["n"], t)
            orden += 1
            if t["aceptada"]:
                texto = f"Aceptada: C real {t['C']:.3f} < {c_actual:.3f} (malla actual) → nueva malla {t['filas']}x{t['columnas']}"
                if t.get("fase") == "tipos":
                    texto = f"Aceptada: C real {t['C']:.3f} < {c_actual:.3f} → el PE pasa a {t['a']}"
                cuadros.append({"tipo": "aceptada", "ep": ep["n"], "rect": anclar(t["estado_rect"]), "anterior": rect,
                                "malla": t.get("malla"), "bloque": t["bloque"], "fase": t.get("fase"),
                                "punto": punto, "texto": texto})
                rect, c_actual, actual = anclar(t["estado_rect"]), t["C"], t.get("malla")
            else:
                c_txt = "-" if t["C"] is None else f"{t['C']:.3f}"
                texto = (f"Descartada: C real {c_txt} ≥ {c_actual:.3f} (malla actual)" if t["estado"] == "correcta"
                         else f"Descartada: {t['estado']}")
                cuadros.append({"tipo": "descartada", "ep": ep["n"], "rect": rect, "malla": actual,
                                "candidato": anclar(t["estado_rect"]), "candidato_malla": t.get("malla"),
                                "bloque": t["bloque"], "fase": t.get("fase"), "punto": punto, "texto": texto})
    return cuadros


def gif_exploracion(datos, ruta, ms=None):
    """El GIF de la exploración por bloques sobre la malla de PEs (ver ``cuadros_exploracion``)."""
    ms = ms or {"inicio": 1800, "control": 1000, "propuesta": 900, "aceptada": 1300, "descartada": 1300}
    malla = MallaPEs(datos["malla_referencia"])
    cuadros = cuadros_exploracion(datos)
    puntos = [c["punto"] for c in cuadros if c.get("punto")]
    todos = [p[2]["C"] for p in puntos if p[2]["C"] is not None]
    mejor = mejor_paso(datos)
    imagenes, duraciones, vistos = [], [], []
    with plt.rc_context(ESTILO):
        fig = plt.figure(figsize=(12, 6), dpi=90)
        ax_m = fig.add_axes([0.02, 0.14, 0.42, 0.74])
        ax_c = fig.add_axes([0.52, 0.14, 0.45, 0.74])
        for k, c in enumerate(cuadros + [{"tipo": "final"}]):
            ax_m.clear()
            ax_c.clear()
            for t in list(fig.texts):
                t.remove()
            if c["tipo"] == "final":
                # la mejor configuración de toda la exploración, con la posición de cada tipo de PE
                activos = malla.activos(anclar(mejor["estado_rect"]), mejor.get("malla")) if mejor else {}
                ax_m.imshow(malla.imagen(activos), interpolation="nearest")
                malla.contorno(ax_m, activos, COLOR_MEJOR, ancho=2.6)
                titulo = "Resultado de la exploración"
                texto = (f"Mejor configuración: {nombre_malla(mejor['malla'])}, C real = {mejor['C']:.3f} "
                         f"(referencia: {todos[0]:.3f})" if mejor else "Ninguna configuración correcta")
            else:
                activos = malla.activos(c["rect"], c.get("malla"))
                resaltados, color = (), COLOR_PROPUESTA
                if c["tipo"] in ("propuesta", "descartada"):
                    candidato = malla.activos(c["candidato"], c.get("candidato_malla"))
                    # lo que la propuesta quita (bloques) o el PE que cambia de tipo (fase de tipos)
                    resaltados = {k for k in activos if k not in candidato or candidato[k] != activos[k]}
                    color = COLOR_PROPUESTA if c["tipo"] == "propuesta" else COLOR_DESCARTADA
                ax_m.imshow(malla.imagen(activos, resaltados, color), interpolation="nearest")
                malla.contorno(ax_m, activos, "#222222", ancho=1.6)
                if c["tipo"] == "aceptada":
                    malla.contorno(ax_m, activos, COLOR_ACEPTADA, ancho=2.6)
                if resaltados:
                    malla.contorno(ax_m, resaltados, color, estilo="--" if c["tipo"] == "propuesta" else "-", ancho=2.2)
                bloque = (" · fase de tipos (E <-> V)" if c.get("fase") == "tipos" else
                          f" · bloque {c['bloque']}" if c.get("bloque") else "")
                titulo = f"Episodio {c['ep']}{bloque}"
                texto = c["texto"]
                if c.get("punto"):
                    vistos.append(c["punto"])
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            ax_m.set_title(titulo, fontsize=11)
            if activos and len(activos) <= 160:  # malla pequeña: recuadro ampliado con la letra de cada PE
                fs = [f for f, _ in activos]
                arriba = min(fs) > malla.filas / 2
                caja = ax_m.inset_axes([0.52, 0.52, 0.45, 0.45] if arriba else [0.52, 0.03, 0.45, 0.45])
                if c["tipo"] in ("propuesta", "descartada"):
                    malla.ampliacion(caja, activos, resaltados, color)
                else:
                    malla.ampliacion(caja, activos)
            ax_m.legend(handles=[Patch(color=COLORES[x], label=x) for x in COLORES]
                        + [Patch(color=COLOR_QUITADO, label="quitado"), Patch(color=COLOR_PROPUESTA, label="propuesto"),
                           Patch(color=COLOR_DESCARTADA, label="descartado")],
                        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6, fontsize=7.5, frameon=False)
            _eje_costos(ax_c, vistos, vistos[-1] if vistos else None)
            _limites_costos(ax_c, todos or [1.0], len(puntos))
            ax_c.set_title(f"Curva de costos ({len(vistos)} de {len(puntos)} puntos)", fontsize=10)
            fig.text(0.5, 0.035, texto, ha="center", fontsize=11, color="#222222",
                     bbox={"facecolor": "#f4f4f4", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4"})
            fig.suptitle(_titulo_episodios(datos), fontsize=9, color="#555555")
            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            imagenes.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
            duraciones.append(4500 if c["tipo"] == "final" else ms[c["tipo"]])
        plt.close(fig)
    imagenes[0].save(ruta, save_all=True, append_images=imagenes[1:], duration=duraciones, loop=0,
                     optimize=False, disposal=1)
    return len(imagenes)


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
    parser.add_argument("--sin-episodios", action="store_true",
                        help="No dibujar curva_episodios.png ni episodios.gif aunque haya episodios.json")
    args = parser.parse_args(argv)

    datos = leer_episodios(args.corrida)
    if datos and datos.get("modo") == "bloques":
        salida = args.salida or args.corrida
        salida.mkdir(parents=True, exist_ok=True)
        curva_episodios_png(datos, salida / "curva_episodios.png")
        curva_costos_png(datos, salida / "curva_costos.png")
        print(f"{salida / 'curva_episodios.png'}\n{salida / 'curva_costos.png'}")
        if not args.sin_gif:
            n = gif_exploracion(datos, salida / "exploracion.gif")
            print(f"{salida / 'exploracion.gif'} ({n} cuadros)")
        return 0

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
    datos = None if args.sin_episodios else leer_episodios(args.corrida)
    if datos and datos.get("episodios"):
        curva_episodios_png(datos, salida / "curva_episodios.png")
        print(f"{salida / 'curva_episodios.png'}")
        if not args.sin_gif:
            n = gif_episodios(datos, historia, salida / "episodios.gif")
            print(f"{salida / 'episodios.gif'} ({n} cuadros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
