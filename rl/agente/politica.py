"""A06 · Política π(C, E) = A, como la define el enunciado.

El LLM es la política. En cada iteración recibe el costo C = K_A·C_A + K_L·C_L
+ K_R·C_R (con los K fijados desde fuera), el estado E (los cambios que produjo
su última acción: malla y C_A, C_L, C_R y C antes y después, y la recompensa
con la que se la premió o penalizó) y
el mejor candidato actual, y escribe la siguiente acción A: la configuración
de la CGRA (cgra.json: malla, tipos de PE y tamaño). No recibe el historial ni
se entrenan pesos. La regla del mejor candidato también es la del enunciado:
si el costo baja, la acción pasa a ser el mejor candidato; si sube, se descarta.

Interfaz que llama el ciclo (A08), igual para todas las implementaciones::

    proponer(contexto, entrada, id_candidata, dir_salida) -> propuesta | None
    incorporar(recibido)                                  -> None

``entrada`` es ``Memoria.entrada_politica()``: ``{"E": ..., "mejor_candidato": ...}``.

Implementaciones:
* ``PoliticaManual``: escribe el prompt en ``prompts/<id>.md`` y devuelve None;
  el agente (p. ej. una sesión de Claude Code) escribe el cgra.json y se evalúa
  con ``python -m rl evaluar``.
* ``PoliticaComando``: llama a un LLM en modo no interactivo en otro proceso
  (``claude -p``, ``codex exec``...), con la plantilla de corrida.json.
* ``PoliticaGuion``: entrega paquetes ya escritos, en orden (pruebas, demos).
"""

import json
import shutil
import subprocess
from pathlib import Path

from ..comun import ARCHIVO_METADATOS, INSTRUCCIONES_AGENTE, RAIZ_REPO, escribir_json, leer_json, relativa


DESCRIPCION_LATENCIA = {
    "ciclos": "ciclos",
    "ciclos_x_nops": "ciclos·(1 + fracción de NOP), fracción = NOPs/(ciclos·PEs): los PEs ociosos también "
                     "cuestan",
}


def construir_prompt(contexto, entrada, id_candidata, dir_salida):
    """Entrada de la política π(C, E) = A en texto: el costo C, el estado E y el mejor candidato."""
    r = contexto["restricciones"]
    p = contexto["parametros_evaluacion"]
    lineas = [
        f"# Acción {id_candidata}: propone una configuración de CGRA para '{contexto['operacion']}'",
        "",
        "Eres el agente del ciclo RL de la CGRA heterogénea. Tu acción es la configuración de la CGRA: tamaño de "
        "malla (filas x columnas), tipos de PE y mapa de distribución. El evaluador (Skills 01–09 + simulador "
        "Python) recibe tu cgra.json junto con el .c, genera los programas de los PEs, los ejecuta y te devuelve "
        "los costos. Tú no escribes programas ni mapeas operaciones. Sigue las instrucciones de "
        f"`{relativa(INSTRUCCIONES_AGENTE)}`.",
        "",
        "## Problema",
        f"- Programa C: `{contexto['programa_c']}` · operación `{contexto['operacion']}` · "
        f"parámetros {json.dumps(contexto['parametros_instancia'])}",
        f"- Regiones de memoria (filas x columnas): {json.dumps(contexto.get('regiones_memoria', {}))}",
    ]
    if contexto.get("operaciones"):
        lineas.append(f"- **Una sola CGRA para todas las operaciones** ({', '.join(contexto['operaciones'])}): "
                      + contexto["evaluacion_conjunto"] + ". En el historial, `por_operacion` da los ciclos de cada una.")
    lineas += [
        "",
        "## Restricciones de la corrida",
        f"- filas {r['filas']} · columnas {r['columnas']} · palabras_memoria <= {r.get('palabras_memoria_max')} · "
        f"ancho_vector en {r.get('anchos_vector_permitidos')}",
        f"- Memoria en los bordes: {r.get('memoria_solo_en_bordes_oeste_este')} (si es True: las columnas "
        "oeste y este son todas M y el interior es solo cómputo; tú eliges qué tipo de cómputo, E, V o C, "
        "va en cada PE del interior) · "
        f"requiere memoria y cómputo: {r.get('requiere_memoria_y_computo')}",
        "- Tipos e instrucciones (copia exactamente estas listas en `tipos_pe`; no inventes tipos ni instrucciones):",
    ]
    for letra, instrucciones in r["instrucciones_por_tipo"].items():
        lineas.append(f"  - {letra} ({r['nombres_tipo'][letra]}): {', '.join(instrucciones)}")
    lineas += [
        f"- Reglas de malla y comunicación: `{r['reglas_malla']}`",
        "",
        "## Política π(C, E) = A",
        "Decides la acción A (la configuración) a partir del costo C y del estado E. El objetivo es minimizar C.",
        "",
        "### Costo C (fijo para la corrida; no lo puedes cambiar)",
        f"- C_A = w1·n_MS + w2·n_MV + w3·n_CS + w4·n_CV con w = {json.dumps(p['w'])} "
        "(M aporta MS+MV, E: CS, V: CV, C: CS+CV, G: las cuatro)",
        f"- C_L = {DESCRIPCION_LATENCIA.get(p['formula_latencia'], p['formula_latencia'])} · C = K_A·C_A/C_A,ref + K_L·C_L/C_L,ref + K_R·C_R con K = {json.dumps(p['K'])}",
        f"- La política premia o penaliza cada acción: r = C_best_prev − C si es correcta; "
        f"r = −{p['penalizacion']['P_fallo']} si es inválida, falla o es incorrecta.",
        "- Si tu acción da un costo menor que el del mejor candidato, pasa a ser el mejor candidato; si da un costo "
        "mayor, se descarta.",
        "",
        "### Estado E: cambios que produjo tu última acción",
        "Comparada con el mejor candidato del que partió (`partio_de`): la malla antes y después, y C_A, C_L, C_R "
        "y C antes, después y su cambio (negativo = bajó). `r` es la recompensa con la que se premió o penalizó.",
        "```json",
        json.dumps(entrada["E"] if entrada else None, indent=1, ensure_ascii=False),
        "```",
        "",
        "### Mejor candidato actual",
        "```json",
        json.dumps(entrada["mejor_candidato"] if entrada else None, indent=1, ensure_ascii=False),
        "```",
        "",
        "## Qué entregar",
        f"Escribe en `{relativa(dir_salida)}/` (y en ningún otro lugar):",
        "- `cgra.json`: formato de `FormatoCGRA` con `filas` y `columnas` enteros, `palabras_memoria`, "
        "`ancho_vector`, `tipos_pe` (solo los que uses, con sus listas exactas) y `malla` explícita: una cadena "
        "por fila, una letra por PE (p. ej. [\"M E M\", \"M V M\"]).",
        f"- `candidata.json`: {{\"id_candidata\": \"{id_candidata}\", \"justificacion\": \"una frase basada en "
        "el costo C, el estado E y el mejor candidato\"}",
        "",
        "Decide solo con la información de este prompt: no leas el historial, los reportes de evaluación ni otras "
        "candidatas. No escribas PE*.txt ni ejecutes el evaluador, y no edites historial, reportes, métricas ni "
        "recompensas. No se garantiza que la próxima acción mejore.",
    ]
    return "\n".join(lineas) + "\n"


def _propuesta(id_candidata, dir_salida):
    metadatos = Path(dir_salida) / ARCHIVO_METADATOS
    justificacion = None
    if metadatos.is_file():
        try:
            justificacion = leer_json(metadatos).get("justificacion")
        except (json.JSONDecodeError, AttributeError):
            justificacion = None
    return {"id_candidata": id_candidata, "justificacion": justificacion, "directorio": str(dir_salida)}


class Politica:
    """Interfaz común. ``incorporar`` no hace falta: la entrada de cada decisión es π(C, E)."""

    def proponer(self, contexto, entrada, id_candidata, dir_salida):
        raise NotImplementedError

    def incorporar(self, recibido):
        return None


class PoliticaManual(Politica):
    """Deja el prompt listo para que el agente (sesión interactiva) escriba el paquete."""

    def __init__(self, carpeta_prompts):
        self.carpeta = Path(carpeta_prompts)
        self.ultimo_prompt = None

    def proponer(self, contexto, entrada, id_candidata, dir_salida):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.ultimo_prompt = self.carpeta / f"{id_candidata}.md"
        self.ultimo_prompt.write_text(construir_prompt(contexto, entrada, id_candidata, dir_salida),
                                      encoding="utf-8")
        return None


class PoliticaComando(Politica):
    """Llama a un LLM en otro proceso. ``comando``: lista con {prompt}, {prompt_archivo}, {dir_salida}, {id}."""

    def __init__(self, comando, carpeta_prompts, timeout_s=1800):
        if not comando:
            raise ValueError("politica.comando vacío")
        self.comando = list(comando)
        self.carpeta = Path(carpeta_prompts)
        self.timeout_s = timeout_s

    def proponer(self, contexto, entrada, id_candidata, dir_salida):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        texto = construir_prompt(contexto, entrada, id_candidata, dir_salida)
        archivo = self.carpeta / f"{id_candidata}.md"
        archivo.write_text(texto, encoding="utf-8")
        valores = {"prompt": texto, "prompt_archivo": str(archivo), "dir_salida": str(dir_salida),
                   "id": id_candidata}
        comando = [parte.format(**valores) for parte in self.comando]
        Path(dir_salida).mkdir(parents=True, exist_ok=True)
        try:
            proceso = subprocess.run(comando, cwd=RAIZ_REPO, capture_output=True, text=True, timeout=self.timeout_s)
            salida = {"codigo": proceso.returncode, "stdout": proceso.stdout[-4000:], "stderr": proceso.stderr[-4000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            salida = {"codigo": None, "error": str(error)}
        escribir_json(self.carpeta / f"{id_candidata}.salida.json", salida)
        propuesta = _propuesta(id_candidata, dir_salida)
        propuesta["salida_politica"] = salida
        return propuesta


class PoliticaGuion(Politica):
    """Entrega, en orden, paquetes ya escritos (copiándolos a ``dir_salida``)."""

    def __init__(self, paquetes):
        self.paquetes = [Path(p) for p in paquetes]
        self.recibidos = []

    def proponer(self, contexto, entrada, id_candidata, dir_salida):
        if not self.paquetes:
            return None
        origen = self.paquetes.pop(0)
        shutil.copytree(origen, dir_salida)
        return _propuesta(id_candidata, dir_salida)

    def incorporar(self, recibido):
        self.recibidos.append(recibido)


def crear_politica(config, carpeta_corrida, guion=None):
    tipo = config.get("tipo", "manual")
    prompts = Path(carpeta_corrida) / "prompts"
    if tipo == "manual":
        return PoliticaManual(prompts)
    if tipo == "comando":
        return PoliticaComando(config.get("comando"), prompts, config.get("timeout_s", 1800))
    if tipo == "guion":
        return PoliticaGuion(guion or config.get("paquetes", []))
    raise ValueError(f"tipo de política desconocido: {tipo}")
