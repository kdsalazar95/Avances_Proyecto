"""E02 · Skills 01–09: de la configuración del agente (cgra.json) + el .c a los PE*.txt.

Es la parte del evaluador "que ya estaba hecha" (diagrama de Evaluación del
PDF): los skills leen las entradas, generan IR y grafos, mapean, calendarizan
y escriben un PE*.txt por PE. Corren en una **sesión de LLM separada** de la
del agente, sin acceso a su historial ni a su recompensa.

Interfaz::

    compilar(ruta_cgra, instancia, dir_salida, errores_previos) -> resultado

``resultado``: ``{"ok", "infraestructura", "detalle", ...}``. ``ok=False`` con
``infraestructura=False`` equivale a "Reportar" del diagrama: los skills no
pudieron dejar los .txt. ``infraestructura=True`` es un problema del entorno
(no hay LLM, se cayó el proceso), no atribuible a la configuración.

Implementaciones:
* ``CompiladorSkills`` (``"tipo": "comando"``): llama al LLM con los skills.
* ``CompiladorReduccion`` (``"tipo": "reduccion_mcm"``): sustituto determinista
  para pruebas y corridas sin LLM (solo reducción en mallas de filas 'M X M' / 'M X',
  X en E, V, C).
"""

import json
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from ..comun import RAIZ_REPO, SKILLS_01_09, escribir_json, relativa

ARCHIVO_REPORTE_SKILLS = "reporte_skills.txt"


def prompt_skills(ruta_cgra, instancia, dir_salida, errores_previos=None):
    lineas = [
        f"# Compilar '{instancia['operacion']}' para la CGRA `{relativa(ruta_cgra)}`",
        "",
        f"Aplica el pipeline de `{relativa(SKILLS_01_09)}/README.md` (skills 01 a 09 y "
        f"`{relativa(SKILLS_01_09 / 'mesh-compatibility.md')}`) con estas diferencias para esta invocación:",
        "",
        f"- Compila **solo** la operación `{instancia['operacion']}` de `{relativa(instancia['fuente_c'])}` "
        f"(parámetros {json.dumps(instancia['parametros_instancia'])}).",
        f"- La CGRA es `{relativa(ruta_cgra)}`, no `cgra/src/cgra.json`. No modifiques ningún cgra.json.",
        f"- La memoria inicial ya existe: `{relativa(instancia['memoria_inicial'])}` (regiones "
        f"{json.dumps(instancia.get('regiones_memoria', {}))}). No la regeneres ni la sobrescribas.",
        f"- Escribe un PE*.txt por PE en `{relativa(Path(dir_salida) / 'pe_instructions')}/` y el manifest de la "
        f"asignación en `{relativa(Path(dir_salida) / 'manifest.json')}`, con al menos las claves `filas`, "
        f"`columnas` (enteros) y `malla` (una cadena por fila, igual a la del cgra.json). No escribas fuera de "
        f"`{relativa(dir_salida)}/`.",
        "- Valida con el skill 09 antes de terminar. No ejecutes run_cgra.py: el evaluador lo hará.",
        f"- No hay nadie a quien preguntar. **Solo** si los datos no coinciden o no puedes generar un programa "
        f"completo, no escribas PE*.txt: escribe el motivo en "
        f"`{relativa(Path(dir_salida) / ARCHIVO_REPORTE_SKILLS)}` y termina. Si compilaste bien, no crees ese "
        f"archivo (el resumen va en el manifest).",
    ]
    if errores_previos:
        lineas += ["", "## Intento anterior: la validación encontró estos errores; corrígelos", ""]
        lineas += [f"- {e}" for e in errores_previos[:30]]
    return "\n".join(lineas) + "\n"


def correr_con_log(comando, cwd, timeout_s, ruta_log, eco=False, titulo=None, etiqueta=""):
    """Corre ``comando`` y escribe su salida (stdout y stderr juntos) en ``ruta_log`` mientras se produce.

    Cada línea del log empieza con ``[mm:ss]``, el tiempo desde el inicio, para poder seguirla en vivo
    (``tail -f``) o convertirla en un GIF (``python -m rl gif``). Con ``eco`` también se muestra en la
    terminal. Devuelve ``(codigo, ultimas_lineas)``; lanza ``FileNotFoundError`` si el comando no existe
    y ``subprocess.TimeoutExpired`` si supera ``timeout_s``.
    """
    ruta_log = Path(ruta_log)
    ultimas = deque(maxlen=200)
    inicio = time.monotonic()
    proceso = subprocess.Popen(comando, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, bufsize=1, errors="replace")
    with ruta_log.open("w", encoding="utf-8") as log:
        def escribir(texto):
            t = int(time.monotonic() - inicio)
            linea = f"[{t // 60:02d}:{t % 60:02d}] {texto}"
            log.write(linea + "\n")
            log.flush()
            if eco:
                print(f"  │ {etiqueta}{linea}", flush=True)

        if titulo:
            escribir(f"== {titulo} ==")

        def leer():
            for linea in proceso.stdout:
                linea = linea.rstrip("\n")
                ultimas.append(linea)
                escribir(linea)

        lector = threading.Thread(target=leer, daemon=True)
        lector.start()
        try:
            codigo = proceso.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proceso.kill()
            proceso.wait()
            lector.join(5)
            escribir(f"== tiempo máximo superado ({timeout_s} s) ==")
            raise
        lector.join(5)
        escribir(f"== terminó con código {codigo} ==")
    return codigo, list(ultimas)


# Tope de sesiones del LLM simultáneas (lotes x operaciones). Se fija con compilador.max_simultaneos.
_SESIONES = threading.BoundedSemaphore(6)


def fijar_max_simultaneos(n):
    global _SESIONES
    _SESIONES = threading.BoundedSemaphore(max(1, int(n)))


class CompiladorSkills:
    """Corre los Skills 01–09 con un LLM en modo no interactivo (``claude -p``, ``codex exec``...).

    La salida del LLM se guarda en vivo en ``eval/<id>/skills/intento-<n>.log`` (ver ``correr_con_log``);
    con ``eco`` también se muestra en la terminal.
    """

    def __init__(self, comando, timeout_s=1800, eco=False):
        if not comando:
            raise ValueError("compilador.comando vacío")
        self.comando = list(comando)
        self.timeout_s = timeout_s
        self.eco = eco

    def compilar(self, ruta_cgra, instancia, dir_salida, errores_previos=None):
        dir_salida = Path(dir_salida)
        dir_salida.mkdir(parents=True, exist_ok=True)
        texto = prompt_skills(ruta_cgra, instancia, dir_salida, errores_previos)
        archivo = dir_salida.parent / f"{dir_salida.name}.prompt.md"
        archivo.write_text(texto, encoding="utf-8")
        valores = {"prompt": texto, "prompt_archivo": str(archivo), "dir_salida": str(dir_salida),
                   "cgra": str(ruta_cgra), "operacion": instancia["operacion"]}
        comando = [parte.format(**valores) for parte in self.comando]
        ruta_log = dir_salida.parent / f"{dir_salida.name}.log"
        print(f"  Skills 01–09 con {Path(comando[0]).name}: trabajando... para verlo en vivo: "
              f"tail -f {relativa(ruta_log)}", file=sys.stderr, flush=True)
        titulo = f"Skills 01–09 · {instancia['operacion']} · {relativa(ruta_cgra)}"
        try:
            with _SESIONES:
                codigo, ultimas = correr_con_log(comando, RAIZ_REPO, self.timeout_s, ruta_log, self.eco, titulo,
                                                 etiqueta=f"{instancia['operacion'][:5]:<5} ")
        except FileNotFoundError as error:
            return {"ok": False, "infraestructura": True, "detalle": f"no se encontró el LLM: {error}"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "infraestructura": False,
                    "detalle": f"los skills superaron el tiempo máximo ({self.timeout_s} s)"}
        salida = "\n".join(ultimas)
        escribir_json(dir_salida.parent / f"{dir_salida.name}.salida.json",
                      {"codigo": codigo, "salida": salida[-4000:], "log": relativa(ruta_log)})
        # El reporte es el "Reportar" del diagrama solo si no dejaron programas; si hay PE*.txt,
        # se tratan como entregados (la validación E03 decide si están bien) y el reporte queda como nota.
        reporte = dir_salida / ARCHIVO_REPORTE_SKILLS
        programas = list((dir_salida / "pe_instructions").glob("PE*.txt"))
        if reporte.is_file() and not programas:
            return {"ok": False, "infraestructura": False,
                    "detalle": "los skills reportaron: " + reporte.read_text(encoding="utf-8").strip()[:2000]}
        if codigo != 0:
            return {"ok": False, "infraestructura": True,
                    "detalle": f"el LLM terminó con código {codigo}: {salida.strip()[-500:]}"}
        return {"ok": True, "infraestructura": False, "detalle": None}


class CompiladorReduccion:
    """Sustituto determinista de los skills: solo reducción en filas 'M X M' / 'M X' (X en E, V, C)."""

    def compilar(self, ruta_cgra, instancia, dir_salida, errores_previos=None):
        from .compilador_sustituto import NoSoportado, compilar
        try:
            n = instancia["parametros_instancia"].get("TAMANO_VECTOR")
            compilar(ruta_cgra, instancia["operacion"], n, dir_salida)
        except NoSoportado as error:
            return {"ok": False, "infraestructura": False, "detalle": f"no se pudo mapear: {error}"}
        return {"ok": True, "infraestructura": False, "detalle": None}


def crear_compilador(config):
    tipo = (config or {}).get("tipo")
    if tipo == "comando":
        if "max_simultaneos" in config:
            fijar_max_simultaneos(config["max_simultaneos"])
        return CompiladorSkills(config.get("comando"), config.get("timeout_s", 1800), config.get("eco", False))
    if tipo == "reduccion_mcm":
        return CompiladorReduccion()
    raise ValueError(f"compilador desconocido: {tipo!r} (use 'comando' o 'reduccion_mcm')")
