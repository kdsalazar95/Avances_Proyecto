"""Rehace todos los informes de una corrida terminada del modo ``bloques`` **sin simular nada**.

Con los datos guardados en la carpeta (``episodios.json``, ``evaluacion_malla.json``,
``evaluaciones_episodios.json`` y ``reporte_corrida.json``) vuelve a escribir:

    configuraciones.csv, configuraciones_operaciones.csv, configuraciones.json   (dijkstra/tabla.py)
    frente_pareto.png                                                           (herramientas/pareto.py)
    curva_costos.png, curva_episodios.png, exploracion.gif                      (herramientas/gif_camino.py)
    reporte_corrida.md, reporte_corrida.json                                    (dijkstra/reporte_corrida.py)

Sirve cuando cambia cómo se presentan los resultados (nombres, gráficas, reporte) y la corrida ya terminó.

    python3 herramientas/regenerar_informes.py resultados/<corrida> [--sin-gif]
"""

import argparse
import sys
from pathlib import Path

RAIZ_DG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_DG))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def regenerar(carpeta, gif=True):
    import gif_camino
    import pareto
    from dijkstra import reporte_corrida
    carpeta = Path(carpeta)
    registros = pareto.frente_de_exploracion(carpeta, carpeta)
    argumentos = [str(carpeta)] + ([] if gif else ["--sin-gif"])
    gif_camino.main(argumentos)
    if (carpeta / "reporte_corrida.json").exists():
        reporte_corrida.regenerar(carpeta)
        print(carpeta / "reporte_corrida.md")
    return registros


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corridas", type=Path, nargs="+", help="Carpetas de resultados de corridas terminadas")
    parser.add_argument("--sin-gif", action="store_true", help="No rehacer exploracion.gif (es lo más lento)")
    args = parser.parse_args(argv)
    for carpeta in args.corridas:
        registros = regenerar(carpeta, gif=not args.sin_gif)
        print(f"{carpeta}: {len(registros)} configuraciones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
