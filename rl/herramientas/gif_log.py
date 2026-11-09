"""Convierte el log de los Skills 01–09 (``eval/<id>/skills/intento-<n>.log``) en un GIF que muestra al LLM trabajando.

El log lo escribe ``rl/evaluador/compilador.py`` mientras el LLM (Codex, Claude...) corre: cada línea
empieza con ``[mm:ss]``. El GIF reproduce esas líneas como una terminal, acelerado: los silencios
largos se acortan para que el GIF dure poco aunque el LLM haya tardado minutos.

    uv run python -m rl gif resultados/corridas/<corrida>/eval/<id>/skills/intento-1.log [-o salida.gif]
"""

import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TIEMPO = re.compile(r"^\[(\d+):(\d{2})\] ?(.*)$")
FUENTES = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf")
FONDO, TEXTO, TENUE, BARRA, ACENTO = "#1e1f24", "#d8dae0", "#7d8290", "#2b2d34", "#e8a25c"


def leer_log(ruta):
    """Lista de ``(segundos, texto)`` del log."""
    eventos = []
    for linea in Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines():
        m = TIEMPO.match(linea)
        if m:
            eventos.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3)))
        elif eventos:
            eventos.append((eventos[-1][0], linea))
    return eventos


def _fuente(tamano):
    for nombre in FUENTES:
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default()


def generar_gif(ruta_log, salida=None, columnas=100, filas=28, tamano=14,
                ms_cuadro=150, ms_max_pausa=700, max_cuadros=180, titulo=None):
    """Escribe el GIF y devuelve su ruta.

    Cada cuadro agrega unas pocas líneas nuevas (las justas para no pasar de ``max_cuadros``) y dura
    ``ms_cuadro``; si hasta el cuadro siguiente pasó tiempo real, se agrega esa pausa, acotada a
    ``ms_max_pausa``. Así un LLM que escribe miles de líneas de golpe no produce un GIF de minutos.
    """
    ruta_log = Path(ruta_log)
    salida = Path(salida) if salida else ruta_log.with_suffix(".gif")
    eventos = leer_log(ruta_log)
    if not eventos:
        raise ValueError(f"{ruta_log}: el log está vacío o no tiene líneas '[mm:ss]'")
    titulo = titulo or ruta_log.parent.name + "/" + ruta_log.name

    # Partir líneas largas; cada cuadro suma ``por_cuadro`` líneas (sin mezclar segundos distintos).
    lineas = [(segundo, parte) for segundo, texto in eventos
              for parte in (textwrap.wrap(texto, columnas, replace_whitespace=False, drop_whitespace=False) or [""])]
    por_cuadro = max(1, -(-len(lineas) // max_cuadros))
    grupos = []
    for segundo, linea in lineas:
        if grupos and grupos[-1][0] == segundo and len(grupos[-1][1]) < por_cuadro:
            grupos[-1][1].append(linea)
        else:
            grupos.append((segundo, [linea]))

    fuente = _fuente(tamano)
    ancho_c = fuente.getbbox("M")[2]
    alto_l = int(tamano * 1.35)
    margen, alto_barra = 12, alto_l + 10
    ancho = margen * 2 + ancho_c * columnas
    alto = alto_barra + margen * 2 + alto_l * filas

    cuadros, duraciones, pantalla = [], [], []
    total = grupos[-1][0]
    for i, (segundo, lineas) in enumerate(grupos):
        pantalla = (pantalla + lineas)[-filas:]
        imagen = Image.new("RGB", (ancho, alto), FONDO)
        dibujo = ImageDraw.Draw(imagen)
        dibujo.rectangle([0, 0, ancho, alto_barra], fill=BARRA)
        for k, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            x = margen + k * 18
            dibujo.ellipse([x, alto_barra // 2 - 5, x + 10, alto_barra // 2 + 5], fill=color)
        dibujo.text((margen + 64, 5), titulo[: columnas - 20], font=fuente, fill=TENUE)
        reloj = f"{segundo // 60:02d}:{segundo % 60:02d} / {total // 60:02d}:{total % 60:02d}"
        dibujo.text((ancho - margen - ancho_c * len(reloj), 5), reloj, font=fuente, fill=ACENTO)
        for j, linea in enumerate(pantalla):
            color = ACENTO if linea.startswith("==") else TEXTO
            dibujo.text((margen, alto_barra + margen + j * alto_l), linea, font=fuente, fill=color)
        cuadros.append(imagen.quantize(colors=8, method=Image.Quantize.MEDIANCUT))
        siguiente = grupos[i + 1][0] if i + 1 < len(grupos) else segundo
        pausa = min(ms_max_pausa, (siguiente - segundo) * 1000)
        duraciones.append(ms_cuadro + pausa)
    duraciones[-1] = 3000  # quedarse en el final
    cuadros[0].save(salida, save_all=True, append_images=cuadros[1:], duration=duraciones, loop=0,
                    optimize=True, disposal=1)
    return salida
