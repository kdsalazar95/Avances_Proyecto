"""Resultados de una corrida además de los datos de convergencia: la tabla de configuraciones, la curva de costos
y el GIF de la exploración (``tabla.py`` y ``gif_corrida.py``). Se escriben al terminar cada corrida por episodios
(salvo con ``"informes": false`` en corrida.json) o con ``python -m rl resultados --corrida DIR``."""

from pathlib import Path


def escribir_resultados(carpeta, gif=True):
    """Escribe configuraciones.csv/.json, curva_costos.png y exploracion.gif; devuelve lo que escribió. Sin
    matplotlib o Pillow escribe solo la tabla y lo avisa."""
    from . import tabla
    carpeta = Path(carpeta)
    escritos = {"tabla": len(tabla.escribir(carpeta))}
    try:
        from . import gif_corrida
        escritos["curva"] = str(gif_corrida.curva_costos_png(carpeta))
        if gif:
            escritos["gif_cuadros"] = gif_corrida.gif_exploracion(carpeta)
    except ImportError as error:
        print(f"Gráficas no generadas ({error}): instale matplotlib y Pillow", flush=True)
    return escritos
