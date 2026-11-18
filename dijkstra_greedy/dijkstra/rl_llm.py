"""Agente RL del enlace epsilon-greedy: un LLM elige el siguiente PE del camino.

El LLM es el CLI que diga ``rl.comando`` en agente.json (por defecto Claude Code CLI,
``claude -p {prompt}``; sirve cualquier CLI que reciba el prompt como argumento y responda por
la salida estándar). En cada paso del camino, con probabilidad ``epsilon``, el LLM recibe el PE
actual con el costo C del camino, y sus vecinos libres con su tipo, su área ``w_tipo`` (de los
pesos w1-w4) y el C que tendría el camino al agregar cada uno (estimado), y además los últimos
episodios y el mejor, con su C **real** (de simular el rectángulo del camino): esa es la
retroalimentación del evaluador. Responde con el identificador de uno de los vecinos. En el
punto de paro (el camino ya cumple los tipos y, según el estimado, ningún vecino baja C), el LLM
decide en cambio si el camino para (``PARAR``) o sigue ``n`` PEs más (``SEGUIR n``). En el modo
``bloques`` (``decidir_bloque``), el LLM elige qué quitar de la malla actual (``FILAS INICIO``,
``FILAS FIN``, ``COLUMNAS INICIO``, ``COLUMNAS FIN``), o ``BAJAR`` (bloque más pequeño) o ``PARAR``. El comando se ejecuta sin shell, con el prompt como argumento (``{prompt}``).
"""

import re
import subprocess
import tempfile

from malla import Malla

# La respuesta válida es solo un identificador de PE; se toleran espacios, comillas invertidas
# alrededor y un punto final ("PE01", "`PE01`", "PE01.").
PATRON_RESPUESTA = re.compile(r"\s*`*\s*(PE\d+)\s*`*\s*\.?\s*")
# En el punto de paro: solo "PARAR" o "SEGUIR n" (sin importar mayúsculas), con la misma tolerancia.
PATRON_PARADA = re.compile(r"\s*`*\s*(?:(PARAR)|SEGUIR\s+(\d+))\s*`*\s*\.?\s*", re.IGNORECASE)


def _num(valor):
    return "-" if valor is None else f"{valor:.6g}"


def _linea_episodio(ep):
    if ep.get("modo") == "bloques":
        return (f"ep {ep['n']}: descenso desde la malla de referencia hasta {ep['filas']}x{ep['columnas']} "
                f"({ep.get('simuladas', 0)} mallas simuladas, {ep.get('aceptadas', 0)} aceptadas), {ep['estado']}, "
                f"C real {_num(ep['C'])}, R = {_num(ep['recompensa'])}")
    marca = " (repetido)" if ep.get("repetido") else ""
    if ep.get("extension"):
        marca += f", siguió +{ep['extension']} PEs después del punto de paro"
    return (f"ep {ep['n']}: camino {'→'.join(ep['camino'])}; rectángulo {' / '.join(ep['malla'])} "
            f"({ep['filas']}x{ep['columnas']}), {ep['estado']}{marca}, C real {_num(ep['C'])} "
            f"(estimado {_num(ep['C_estimado'])}), R = {_num(ep['recompensa'])}")


def bloque_historial(contexto):
    """El texto de los episodios anteriores para el prompt ("" si todavía no hay ninguno)."""
    if not contexto or not contexto.get("historial"):
        return ""
    lineas = "\n".join(f"- {_linea_episodio(ep)}" for ep in contexto["historial"])
    mejor = contexto.get("mejor")
    texto_mejor = (f"Mejor hasta ahora: {_linea_episodio(mejor)}" if mejor else
                   "Todavía ningún episodio mejoró el C de la malla completa.")
    return (
        f"Episodio {contexto['episodio']}. Episodios anteriores: cada camino se convirtió en el rectángulo "
        "que lo contiene y se simuló; su C es el real (el C de los vecinos de abajo es un estimado). "
        f"R = C_best anterior − C real (positiva si mejoró). C de la malla completa: {_num(contexto.get('C_malla'))}.\n"
        f"{lineas}\n{texto_mejor}\n"
        "Usa estos resultados para elegir: evita repetir caminos que no mejoraron.\n"
    )


def construir_prompt_parada(actual, camino, c_actual, contexto, maximo):
    """El prompt del punto de paro: el camino ya cumple los tipos y, según el estimado, ningún vecino
    baja C. El LLM decide PARAR o SEGUIR n (1 ≤ n ≤ ``maximo``), con el historial de episodios (C real)
    y el rectángulo que se simularía si para ahora (``contexto["rectangulo"]``)."""
    rectangulo = contexto.get("rectangulo") if contexto else None
    texto_rect = (f"Si para ahora, se simula el rectángulo {' / '.join(rectangulo)} "
                  f"({len(rectangulo)}x{len(rectangulo[0].split())}).\n" if rectangulo else "")
    return (
        "Eres el agente de RL que decide el tamaño de una configuración de una CGRA, a partir de una malla de "
        "referencia (la malla completa).\n"
        "El objetivo es minimizar el costo C = promedio sobre las operaciones de "
        "K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R (C_A: área, suma de w_tipo de los PEs; C_L: latencia; "
        "C_R: error).\n"
        f"{bloque_historial(contexto)}"
        f"El camino va en {actual}, lleva {len(camino)} PEs ({'→'.join(camino)}), ya tiene PEs de memoria (M), "
        f"escalar (E) y vectorial (V), y su C estimado es {c_actual:.6g}. Según el estimado, agregar cualquier "
        "vecino sube C, así que el greedy pararía aquí. El estimado usa la latencia de la malla completa y no "
        "cambia al quitar PEs; el C real de los episodios anteriores (de simular su rectángulo) sí la refleja.\n"
        f"{texto_rect}"
        f"Decide: PARAR (el episodio termina con este camino) o SEGUIR n (agregar n PEs más, con n entre 1 y "
        f"{maximo}; después se vuelve a revisar).\n"
        "Responde solo con PARAR o con SEGUIR n, por ejemplo SEGUIR 3."
    )


def interpretar_parada(salida, maximo):
    """En el punto de paro: 0 si la respuesta es solo ``PARAR``; ``n`` si es solo ``SEGUIR n`` con
    1 ≤ n ≤ ``maximo``; ``None`` si no (más texto, n fuera de rango): el agente lo toma como parar."""
    coincidencia = PATRON_PARADA.fullmatch(salida or "")
    if not coincidencia:
        return None
    if coincidencia.group(1):
        return 0
    n = int(coincidencia.group(2))
    return n if 1 <= n <= maximo else None


ACCIONES_BLOQUE = ("FILAS INICIO", "FILAS FIN", "COLUMNAS INICIO", "COLUMNAS FIN", "BAJAR", "PARAR")
PATRON_BLOQUE = re.compile(r"\s*`*\s*(FILAS\s+INICIO|FILAS\s+FIN|COLUMNAS\s+INICIO|COLUMNAS\s+FIN|BAJAR|PARAR)"
                           r"\s*`*\s*\.?\s*", re.IGNORECASE)


def _linea_paso(t):
    if t["accion"] in ("BAJAR", "PARAR"):
        return f"- {t['accion']} ({'fase de tipos' if t.get('fase') == 'tipos' else 'bloque ' + str(t['bloque'])})"
    if t.get("fase") == "tipos":
        resultado = "aceptada" if t["aceptada"] else "descartada"
        return (f"- {t['accion']}: PE (fila {t['pe'][0]}, columna {t['pe'][1]}) de {t['de']} a {t['a']}: "
                f"{t['estado']}, C real {_num(t['C'])} (estimado {_num(t['C_estimado'])}), {resultado}")
    if t["accion"] == "inicio":
        return f"- inicio: {t['filas']}x{t['columnas']}, C real {_num(t['C'])}"
    resultado = "aceptada" if t["aceptada"] else "descartada"
    return (f"- {t['accion']} (bloque {t['bloque']}) → {t['filas']}x{t['columnas']}: {t['estado']}, "
            f"C real {_num(t['C'])} (estimado {_num(t['C_estimado'])}), {resultado}")


def construir_prompt_bloques(contexto):
    """El prompt del modo ``bloques``: los episodios anteriores, el descenso del episodio actual con el
    C real de cada malla simulada, la malla actual y las acciones posibles con su C estimado."""
    opciones = "\n".join(f"- {accion}: queda {o['tamano']}, C estimado {_num(o['C_estimado'])}"
                         for accion, o in contexto["opciones"].items())
    pasos = "\n".join(_linea_paso(t) for t in contexto.get("trayectoria", []))
    return (
        "Eres el agente de RL que explora configuraciones de una CGRA reduciendo una malla de referencia "
        "por bloques.\n"
        "El objetivo es minimizar el costo C = promedio sobre las operaciones de "
        "K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R (C_A: área; C_L: latencia; C_R: error). Quitar PEs "
        "reduce el área pero puede aumentar la latencia.\n"
        f"{bloque_historial(contexto)}"
        f"Descenso de este episodio (cada malla se simuló; se acepta si su C real baja):\n{pasos}\n"
        f"Malla actual: {contexto['malla_actual']}, C real {_num(contexto['C_actual'])}. Bloque actual: "
        f"{contexto['bloque']} (tamaños: {', '.join(str(b) for b in contexto['bloques'])}).\n"
        "Acciones (el C estimado no simula: usa la latencia de la malla de referencia y favorece las mallas "
        f"pequeñas; el C real de arriba sí refleja la latencia):\n{opciones}\n"
        "- BAJAR: pasar al bloque siguiente, más pequeño\n"
        "- PARAR: terminar el descenso con la malla actual\n\n"
        "Responde solo con una de las acciones, exactamente como aparece (por ejemplo FILAS FIN)."
    )


def interpretar_bloque(salida, opciones):
    """La acción si la respuesta es **solo** una acción permitida (``opciones``, ``BAJAR`` o ``PARAR``),
    sin importar mayúsculas y con espacios, comillas invertidas o un punto final tolerados; si no, ``None``."""
    coincidencia = PATRON_BLOQUE.fullmatch(salida or "")
    if not coincidencia:
        return None
    accion = " ".join(coincidencia.group(1).upper().split())
    return accion if accion in set(opciones) | {"BAJAR", "PARAR"} else None


PATRON_TIPO = re.compile(r"\s*`*\s*(CAMBIAR\s+\d+\s+\d+|PARAR)\s*`*\s*\.?\s*", re.IGNORECASE)


def construir_prompt_tipos(contexto):
    """El prompt de la fase de tipos: los episodios anteriores, lo probado en este episodio, la malla actual
    (una fila por línea) y los cambios posibles de un PE del interior (E <-> V) con su C estimado."""
    opciones = "\n".join(f"- {accion}: PE (fila {o['pe'][0]}, columna {o['pe'][1]}) de {o['de']} a {o['a']}, "
                         f"C estimado {_num(o['C_estimado'])}" for accion, o in contexto["opciones"].items())
    pasos = "\n".join(_linea_paso(t) for t in contexto.get("trayectoria", []))
    malla = "\n".join(f"  fila {i}: {fila}" for i, fila in enumerate(contexto["malla_actual"]))
    return (
        "Eres el agente de RL que explora configuraciones de una CGRA. Ya se redujo la malla por bloques; "
        "ahora decides el tipo de cada PE del interior: escalar (E) o vectorial (V). Las columnas de los "
        "bordes son de memoria (M) y no cambian.\n"
        "El objetivo es minimizar el costo C = promedio sobre las operaciones de "
        "K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R (C_A: área, un V cuesta más área que un E; C_L: "
        "latencia, las operaciones vectoriales solo corren en PEs V; C_R: error).\n"
        f"{bloque_historial(contexto)}"
        f"Este episodio (cada malla se simuló; se acepta si su C real baja):\n{pasos}\n"
        f"Malla actual (columnas numeradas desde 0), C real {_num(contexto['C_actual'])}:\n{malla}\n"
        "Cambios posibles (el C estimado solo cuenta el cambio de área y supone la misma latencia; el C real "
        f"de arriba sí refleja la latencia):\n{opciones}\n"
        "- PARAR: terminar con la malla actual\n\n"
        "Responde solo con una de las acciones, exactamente como aparece (por ejemplo CAMBIAR 1 2)."
    )


def interpretar_tipo(salida, opciones):
    """La acción si la respuesta es **solo** ``CAMBIAR f c`` de ``opciones`` o ``PARAR``; si no, ``None``."""
    coincidencia = PATRON_TIPO.fullmatch(salida or "")
    if not coincidencia:
        return None
    accion = " ".join(coincidencia.group(1).upper().split())
    return accion if accion in set(opciones) | {"PARAR"} else None


def construir_prompt(actual, libres, pesos, camino, opciones, c_actual, contexto=None):
    """El prompt del paso: los episodios anteriores con su C real (``contexto``, de
    ``Agente.contexto_rl``), el PE actual con el C del camino, y los vecinos libres con su tipo, su
    área ``w_tipo`` y el C del camino si se agrega cada uno. ``opciones``: ``{PE: {"C": C efectivo,
    "cumple_tipos": bool}}`` (lo arma el agente); ``c_actual``: el C efectivo del camino actual."""
    filas = "\n".join(
        f"- {pe}: tipo {pesos[pe].tipo}, área w_tipo {pesos[pe].w_tipo:g}, C del camino si se agrega "
        f"{opciones[pe]['C']:.6g} ({'cumple' if opciones[pe]['cumple_tipos'] else 'no cumple'} los tipos de PE), "
        f"fila {Malla.posicion(pe)[0]}, columna {Malla.posicion(pe)[1]}"
        for pe in libres)
    return (
        "Eres el agente de RL que elige el siguiente PE de un camino por una malla de una CGRA.\n"
        "El objetivo es minimizar el costo C = promedio sobre las operaciones de "
        "K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R (C_A: área, suma de w_tipo de los PEs; C_L: latencia; "
        "C_R: error). Un camino que todavía no tiene PEs de memoria (M), escalar (E) y vectorial (V) "
        "no cumple los tipos de PE y cuesta P_fallo.\n"
        f"{bloque_historial(contexto)}"
        f"El camino va en {actual}, lleva {len(camino)} PEs y su C es {c_actual:.6g}. "
        "Elige uno de los vecinos libres (no visitados); un C menor es mejor.\n"
        f"Vecinos libres:\n{filas}\n\n"
        f"Responde solo con el identificador de uno de ellos, por ejemplo {libres[0]}."
    )


def interpretar(salida, libres):
    """El PE elegido si la respuesta es **solo** el identificador de un vecino libre (con espacios,
    comillas invertidas o un punto final tolerados); ``None`` si no (más texto, varios PEs, un PE
    que no está libre). Así nunca se le atribuye al LLM una elección que no hizo: con "descarto PE10,
    elijo PE01" no se adivina cuál quiso; el paso cae a greedy y se cuenta como ``rl_invalida``."""
    coincidencia = PATRON_RESPUESTA.fullmatch(salida or "")
    if coincidencia and coincidencia.group(1) in libres:
        return coincidencia.group(1)
    return None


class EleccionLLM:
    """El agente RL por CLI: ``eleccion(actual, libres, pesos, camino, opciones, c_actual, contexto) -> PE`` (o
    ``None`` si la respuesta no sirve). ``comando``: la lista de ``rl.comando`` de agente.json, con
    ``{prompt}`` donde va el prompt del paso (por defecto ``["claude", "-p", "{prompt}"]``)."""

    def __init__(self, comando, timeout_s=300):
        if not comando or not any("{prompt}" in parte for parte in comando):
            raise ValueError("rl.comando debe incluir {prompt} (el prompt del paso)")
        self.comando = list(comando)
        self.timeout_s = timeout_s
        self.registro = []  # cada consulta: tipo, segundos, timeout, válida, respuesta (para el reporte)

    def __call__(self, actual, libres, pesos, camino, opciones, c_actual, contexto=None, parada=None):
        """El PE elegido (o ``None``); con ``parada`` (el máximo de PEs extra), en el punto de paro:
        0 = parar, ``n`` = seguir n PEs más, ``None`` = respuesta inválida."""
        if parada:
            prompt = construir_prompt_parada(actual, camino, c_actual, contexto, parada)
        else:
            prompt = construir_prompt(actual, libres, pesos, camino, opciones, c_actual, contexto)
        salida = self._consultar(prompt, "paro" if parada else "paso")
        if salida is None:
            return None
        respuesta = interpretar_parada(salida, parada) if parada else interpretar(salida, libres)
        self.registro[-1]["valida"] = respuesta is not None
        return respuesta

    def decidir_bloque(self, contexto):
        """Modo ``bloques``: la acción elegida (una de ``contexto["opciones"]``, ``BAJAR`` o ``PARAR``), o
        ``None`` si la respuesta no sirve (se usa la elección de Greedy)."""
        salida = self._consultar(construir_prompt_bloques(contexto), "bloque")
        if salida is None:
            return None
        respuesta = interpretar_bloque(salida, contexto["opciones"])
        self.registro[-1]["valida"] = respuesta is not None
        return respuesta

    def decidir_tipo(self, contexto):
        """Fase de tipos: ``CAMBIAR f c`` (una de ``contexto["opciones"]``) o ``PARAR``, o ``None`` si la
        respuesta no sirve (se usa la elección de Greedy)."""
        salida = self._consultar(construir_prompt_tipos(contexto), "tipo")
        if salida is None:
            return None
        respuesta = interpretar_tipo(salida, contexto["opciones"])
        self.registro[-1]["valida"] = respuesta is not None
        return respuesta

    def _consultar(self, prompt, tipo="paso"):
        """La salida del CLI con el prompt; ``None`` si se agota el tiempo. Registra la consulta."""
        import time
        argumentos = [parte.replace("{prompt}", prompt) for parte in self.comando]
        t0 = time.monotonic()
        entrada = {"tipo": tipo, "segundos": None, "timeout": False, "valida": False, "respuesta": None}
        self.registro.append(entrada)
        try:
            proceso = subprocess.run(argumentos, capture_output=True, text=True, timeout=self.timeout_s,
                                     cwd=tempfile.gettempdir())
        except FileNotFoundError:
            raise RuntimeError(f"no se encontró el comando del agente RL: {self.comando[0]!r}")
        except subprocess.TimeoutExpired:
            entrada.update(segundos=round(time.monotonic() - t0, 2), timeout=True)
            return None
        entrada.update(segundos=round(time.monotonic() - t0, 2), respuesta=(proceso.stdout or "").strip()[:80])
        if proceso.returncode != 0:
            raise RuntimeError(f"el agente RL terminó con código {proceso.returncode}: "
                               f"{(proceso.stderr or proceso.stdout).strip()[:500]}")
        return proceso.stdout
