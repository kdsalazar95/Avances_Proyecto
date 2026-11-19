"""Barrido de epsilon: una corrida por valor de epsilon con la misma configuración y la misma semilla.

    uv run python -m rl barrido --config rl/config/comparacion.json --epsilons 0 10 20 80 90 100

Cada corrida se llama ``<id>-eps<NNN>`` (``eps010`` = epsilon 0.10). La referencia 32x32 se evalúa una sola
vez, en la primera corrida, y las demás reusan esa evaluación (misma malla y mismos normalizadores). Si una
corrida ya existe, se reanuda donde quedó. Al final se escriben los archivos de la comparación
(``herramientas/convergencia.py``) en ``<id>-comparacion/``.
"""

from pathlib import Path

from ..agente.ciclo import Ciclo, ErrorCiclo, iniciar
from ..agente.politica import crear_politica
from ..corrida import cargar_corrida
from .convergencia import comparar


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


def barrido(ruta_config, epsilons, carpeta_corridas, max_iteraciones=None, politica=None, al_terminar=None):
    """Corre (o reanuda) una corrida por epsilon y escribe la comparación. Devuelve ``(carpetas, resumenes)``.

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
            ciclo = iniciar(ruta_config, carpeta_corridas, cambios={"id": carpeta.name, "epsilon": epsilon},
                            referencia_de=primera)
        primera = primera or carpeta
        p = politica(ciclo) if politica else crear_politica(ciclo.corrida["politica"], ciclo.carpeta,
                                                            corrida=ciclo.corrida)
        estado = ciclo.correr(p, max_iteraciones)
        if al_terminar:
            al_terminar(ciclo, estado)
        carpetas.append(carpeta)
    resumenes = comparar(carpetas, carpeta_corridas / f"{base['id']}-comparacion")
    return carpetas, resumenes
