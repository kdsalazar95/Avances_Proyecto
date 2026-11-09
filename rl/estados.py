"""Estados de evaluación (docs/solucion_rl.md, tabla del pseudocódigo).

La acción del agente es solo la configuración de la CGRA (cgra.json). El
evaluador es lo que ya existía: Skills 01–09 (generan los PE*.txt) + el
simulador Python.
"""

ACCION_INVALIDA = "accion_invalida"      # el cgra.json no pasó la validación previa (A03)
FALLO_EVALUADOR = "fallo_evaluador"      # los Skills 01–09 no dejaron .txt válidos tras N_corr intentos ("Reportar")
FALLO_SIMULADOR = "fallo_simulador"      # tiempo excedido, bloqueo, instrucción inválida o error del simulador
INCORRECTA = "incorrecta"                # el simulador terminó, pero el resultado no pasó la validación funcional
CORRECTA = "correcta"
ERROR_INFRAESTRUCTURA = "error_infraestructura"  # no atribuible: falta el LLM/compilador, disco, etc.

TODOS = (ACCION_INVALIDA, FALLO_EVALUADOR, FALLO_SIMULADOR, INCORRECTA, CORRECTA, ERROR_INFRAESTRUCTURA)

# Reciben el costo alto P_fallo y r = -P_fallo.
PENALIZADOS = frozenset({ACCION_INVALIDA, FALLO_EVALUADOR, FALLO_SIMULADOR, INCORRECTA})

# No se atribuyen al agente: sin costo ni recompensa, no cuentan para la paciencia.
NO_ATRIBUIBLES = frozenset({ERROR_INFRAESTRUCTURA})
