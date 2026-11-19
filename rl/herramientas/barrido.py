"""Barrido de epsilon (y de paciencia): una corrida por epsilon con la misma configuración y la misma semilla.

    uv run python -m rl barrido --config rl/config/comparacion.json --epsilons 0 10 20 80 90 100

Cada corrida se llama ``<id>-eps<NNN>`` (``eps010`` = epsilon 0.10) y va por episodios con la mayor paciencia
(``paciencia``, 10) y un tope que no la limite (``max_episodios``, 30). Los resultados de paciencia 3 y 6 salen de
cortar esas mismas corridas donde se habrían detenido (``convergencia.corte``), como en el Sistema 2: 6 corridas dan
las 18 combinaciones de epsilon y paciencia.

La referencia 32x32 se evalúa una sola vez, en la primera corrida, y las demás reusan esa evaluación (misma malla y
mismos normalizadores). Las corridas comparten la caché de evaluaciones (``<id>-cache/``): una configuración ya
evaluada en otra corrida no se vuelve a evaluar y cuenta lo que tardó cuando se evaluó. Si una corrida ya existe,
se reanuda donde quedó. Al final se escriben los archivos de la comparación en ``<id>-comparacion/``, con sus gráficas (``graficas.py``).
"""

from pathlib import Path

from ..agente.ciclo import Ciclo, ErrorCiclo, iniciar
from ..agente.politica import crear_politica
from ..corrida import cargar_corrida
from .convergencia import PACIENCIAS, comparar


def leer_epsilons(valores):
    """``["0", "10", "0.2"]`` -> ``[0.0, 0.1, 0.2]``: los valores mayores que 1 son porcentajes."""
    epsilons = []
    for valor in valores:
        e = float(valor)
        e = e / 100 if e > 1 else e
        if not 0 <= e <= 1:
            raise ValueError(f"epsilon fuera de rango: {valor}")
        epsilons.append(round(e, 6))
    return epsilons


def nombre(id_base, epsilon):
    return f"{id_base}-eps{round(epsilon * 100):03d}"


def barrido(ruta_config, epsilons, carpeta_corridas, max_iteraciones=None, politica=None, al_terminar=None,
            paciencia=10, max_episodios=30, paciencias=None):
    """Corre (o reanuda) una corrida por epsilon y escribe la comparación. Devuelve ``(carpetas, resumenes)``.
    ``paciencias``: las que se calculan cortando cada corrida (por defecto ``convergencia.PACIENCIAS``).

    ``politica(ciclo)``: fábrica de la política (por defecto, la de corrida.json envuelta con su epsilon).
    ``al_terminar(ciclo, estado)``: se llama después de cada corrida (para mostrar el avance)."""
    base = cargar_corrida(ruta_config)
    if politica is None and base["politica"].get("tipo", "manual") == "manual" and any(e < 1 for e in epsilons):
        raise ErrorCiclo("el barrido corre solo: la política de la configuración debe ser 'comando' (el LLM), "
                         "no 'manual'")
    carpeta_corridas = Path(carpeta_corridas)
    carpetas, primera = [], None
    for epsilon in epsilons:
        carpeta = carpeta_corridas / nombre(base["id"], epsilon)
        if (carpeta / "corrida.json").is_file():
            ciclo = Ciclo(carpeta)
        else:
            limites = {**base["limites"], "paciencia": paciencia, "max_episodios": max_episodios}
            cambios = {"id": carpeta.name, "epsilon": epsilon, "limites": limites,
                       "cache_evaluaciones": str(carpeta_corridas / f"{base['id']}-cache")}
            ciclo = iniciar(ruta_config, carpeta_corridas, cambios=cambios, referencia_de=primera)
        primera = primera or carpeta
        p = politica(ciclo) if politica else crear_politica(ciclo.corrida["politica"], ciclo.carpeta,
                                                            corrida=ciclo.corrida)
        estado = ciclo.correr(p, max_iteraciones)
        if al_terminar:
            al_terminar(ciclo, estado)
        carpetas.append(carpeta)
    salida = carpeta_corridas / f"{base['id']}-comparacion"
    resumenes = comparar(carpetas, salida, paciencias or PACIENCIAS)
    if base.get("informes", True):
        try:
            from .graficas import graficar
            graficar([salida], salida)
        except ImportError as error:
            print(f"Gráficas de la comparación no generadas ({error}): instale matplotlib", flush=True)
    return carpetas, resumenes
