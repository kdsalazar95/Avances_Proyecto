"""E05 · Validación funcional elemento por elemento.

Criterio para cada elemento i:  |y_i - ŷ_i| <= eps_abs + eps_rel * |y_i|.
La acción es correcta solo si todos los elementos cumplen. Además se mide
C_R = promedio de |y_i - ŷ_i| / (|y_i| + epsilon_cr).
"""

import math

from ..comun import usar_simulador

usar_simulador()
from memoria import ImagenMemoria  # noqa: E402


def comparar(esperado, obtenido, formas_obtenidas, eps_abs, eps_rel, epsilon_cr, max_fallos=10):
    """Compara regiones esperadas con las obtenidas.

    ``esperado``: ``{region: {"forma": [f, c], "valores": [...]}}``;
    ``obtenido``: ``{region: [valores]}``; ``formas_obtenidas``: ``{region: (f, c)}``.
    """
    total, fallidos, suma_error = 0, 0, 0.0
    primeros, estructurales = [], []
    for nombre, datos in esperado.items():
        if nombre not in obtenido:
            estructurales.append(f"falta la región '{nombre}' en la memoria final")
            continue
        if list(formas_obtenidas[nombre]) != list(datos["forma"]) or len(obtenido[nombre]) != len(datos["valores"]):
            estructurales.append(f"'{nombre}': forma {list(formas_obtenidas[nombre])} y se esperaba {datos['forma']}")
            continue
        for i, (y, y_hat) in enumerate(zip(datos["valores"], obtenido[nombre])):
            total += 1
            if not math.isfinite(y_hat):
                fallidos += 1
                suma_error = math.inf
                if len(primeros) < max_fallos:
                    primeros.append({"region": nombre, "i": i, "esperado": y, "obtenido": str(y_hat)})
                continue
            diferencia = abs(y - y_hat)
            suma_error += diferencia / (abs(y) + epsilon_cr)
            if diferencia > eps_abs + eps_rel * abs(y):
                fallidos += 1
                if len(primeros) < max_fallos:
                    primeros.append({"region": nombre, "i": i, "esperado": y, "obtenido": y_hat})
    aprobada = not estructurales and fallidos == 0 and total > 0
    c_r = None if estructurales or total == 0 else suma_error / total
    if c_r is not None and not math.isfinite(c_r):
        c_r = None
    return {
        "aprobada": aprobada,
        "total_elementos": total,
        "elementos_fallidos": fallidos,
        "primeros_fallos": primeros,
        "errores_estructurales": estructurales,
        "C_R": c_r,
        "estado": "correcta" if aprobada else "incorrecta",
    }


def validar_memoria(ruta_memoria_final, esperado, tolerancias):
    """Lee la memoria final del simulador y la compara con el esperado."""
    regiones, formas = ImagenMemoria.leer(ruta_memoria_final).regiones()
    return comparar(esperado, regiones, formas, tolerancias["eps_abs"], tolerancias["eps_rel"],
                    tolerancias["epsilon_cr"])
