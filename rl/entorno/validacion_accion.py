"""A03 · Validación previa de la acción del agente (su cgra.json).

La acción es la configuración de la CGRA: tamaño de malla, tipos de PE y mapa
de distribución. Antes de mandarla al evaluador se comprueba que sea coherente
(docs/solucion_rl.md, "Validación previa"):

- el cgra.json existe, está dentro de la candidata y lo acepta ``FormatoCGRA``;
- filas, columnas, memoria y ancho vectorial dentro de los límites de la corrida;
- los tipos y sus instrucciones son exactamente los del esquema real;
- el mapa coincide con el tamaño (lo verifica ``FormatoCGRA``);
- hay al menos un PE de memoria y uno de cómputo;
- si la corrida lo exige, las columnas oeste y este son todas de memoria (M)
  y el interior es solo de cómputo: qué tipo de cómputo (E, V, C) va en cada
  PE del interior lo decide el agente.

Así no llegan al Skill 01 configuraciones cuyos datos no coinciden. No corrige
nada: una corrección es una nueva propuesta.
"""

import json
from pathlib import Path

from ..comun import ARCHIVO_CONFIG, CARPETA_PROGRAMAS, usar_simulador

usar_simulador()
from formato_cgra import FormatoCGRA  # noqa: E402

MEMORIA_OPS = {"LD", "ST", "VLD", "VST"}
COMPUTO_OPS = {"ADD", "SUB", "MUL", "DIV", "VADD", "VSUB", "VMUL", "VDIV"}


def _dentro(ruta, base):
    try:
        Path(ruta).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def _motivos_bordes(formato):
    """Memoria en los bordes: oeste y este son M; el interior, cómputo sin acceso a memoria."""
    if formato.columnas < 3:
        return [f"con {formato.columnas} columna(s) no hay interior para el cómputo: la memoria va en las "
                "columnas oeste y este y el cómputo entre ellas (se necesitan al menos 3 columnas)"]
    motivos = []
    bordes = (0, formato.columnas - 1)
    for f, fila in enumerate(formato.malla):
        for c, letra in enumerate(fila):
            accede_memoria = bool(set(formato.tipos_pe[letra]["instrucciones"]) & MEMORIA_OPS)
            if c in bordes and letra != "M":
                motivos.append(f"PE({f},{c}) es {letra}: los bordes oeste y este deben ser de memoria (M)")
            elif c not in bordes and accede_memoria:
                motivos.append(f"PE({f},{c}) es {letra} y accede a memoria: la memoria solo puede ir en los "
                               "bordes oeste/este; el interior es de cómputo (E, V o C)")
    return motivos


def validar_accion(directorio, contexto, id_candidata=None):
    """Devuelve ``{"id_candidata", "paquete_valido", "motivos"}``."""
    directorio = Path(directorio)
    if not directorio.is_dir():
        return {"id_candidata": id_candidata, "paquete_valido": False, "motivos": [f"no existe {directorio}"]}
    motivos = []
    ruta = directorio / ARCHIVO_CONFIG
    if not ruta.is_file():
        return {"id_candidata": id_candidata, "paquete_valido": False, "motivos": ["falta cgra.json"]}
    if ruta.is_symlink() or not _dentro(ruta, directorio):
        motivos.append("cgra.json es un enlace o apunta fuera de la candidata")
    if (directorio / CARPETA_PROGRAMAS).exists():
        motivos.append("la acción del agente es solo la configuración: los PE*.txt los generan los Skills 01–09 "
                       "del evaluador (quite pe_instructions/)")

    # Esquema real de FormatoCGRA (incluye: el mapa coincide con filas x columnas).
    try:
        formato = FormatoCGRA.cargar(ruta)
    except ValueError as error:
        return {"id_candidata": id_candidata, "paquete_valido": False, "motivos": motivos + [f"cgra.json: {error}"]}
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    r = contexto["restricciones"]
    for eje, valor in (("filas", formato.filas), ("columnas", formato.columnas)):
        minimo, maximo = r[eje].get("minimo", 1), r[eje].get("maximo")
        if valor < minimo or (maximo is not None and valor > maximo):
            motivos.append(f"{eje} = {valor} fuera del rango permitido [{minimo}, {maximo}]")
    if r.get("palabras_memoria_max") is not None and formato.palabras_memoria > r["palabras_memoria_max"]:
        motivos.append(f"palabras_memoria = {formato.palabras_memoria} supera {r['palabras_memoria_max']}")
    if formato.ancho_vector is not None and r.get("anchos_vector_permitidos") and \
            formato.ancho_vector not in r["anchos_vector_permitidos"]:
        motivos.append(f"ancho_vector = {formato.ancho_vector} no está en {r['anchos_vector_permitidos']}")
    if "tipos_pe" not in datos:
        motivos.append("cgra.json debe declarar 'tipos_pe' (tomados del esquema real)")

    # Tipos e instrucciones: solo los del esquema real.
    reales = r["instrucciones_por_tipo"]
    usados = {letra for fila in formato.malla for letra in fila}
    for letra, tipo in formato.tipos_pe.items():
        if letra not in reales or letra not in r.get("tipos_pe", reales):
            motivos.append(f"tipo de PE '{letra}' no existe en el esquema de la corrida")
        elif set(tipo["instrucciones"]) != {op.upper() for op in reales[letra]}:
            motivos.append(f"el tipo '{letra}' cambia su lista de instrucciones respecto al esquema real")

    # Reglas de topología aprobadas para la corrida.
    if r.get("requiere_memoria_y_computo"):
        ops = set().union(*(formato.tipos_pe[l]["instrucciones"] for l in usados))
        if not ops & MEMORIA_OPS:
            motivos.append("la malla no tiene ningún PE de memoria")
        if not ops & COMPUTO_OPS:
            motivos.append("la malla no tiene ningún PE de cómputo")
    if r.get("memoria_solo_en_bordes_oeste_este"):
        motivos += _motivos_bordes(formato)

    return {"id_candidata": id_candidata, "paquete_valido": not motivos, "motivos": motivos}
