"""Formato compacto compartido para mostrar números."""


def formatear_numero(valor):
    """Redondea números a seis decimales para mostrarlos de forma compacta."""
    return f"{round(valor, 6):g}" if isinstance(valor, (int, float)) else str(valor)
