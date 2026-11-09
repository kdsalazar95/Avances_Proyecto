"""A01 · Contexto estructurado del problema para el agente LLM.

Tipos e instrucciones salen del ``cgra.json`` real (esquema); los límites de
búsqueda y la evaluación salen de corrida.json. El contexto describe lo que el
agente debe entregar: la configuración de la CGRA (malla, tipos de PE, tamaño).
"""

from ..comun import INSTRUCCIONES_AGENTE, SKILLS_01_09, leer_json, relativa
from ..corrida import resolver

ENTREGA = {
    "directorio": "el que indique el ciclo (candidatas/<id_candidata>/)",
    "archivos": {
        "cgra.json": "configuración de la CGRA en el formato de FormatoCGRA: 'filas' y 'columnas' enteros, "
                     "tipos_pe del esquema real y una 'malla' explícita (mapa de distribución)",
        "candidata.json": "id_candidata y justificacion breve basada en el costo C, el estado E y el mejor candidato",
    },
    "no_entregar": "PE*.txt: los generan los Skills 01–09 dentro del evaluador",
}


def limites_esquema(esquema):
    """Rangos de filas/columnas declarados en el esquema (si son rangos)."""
    limites = {}
    for eje in ("filas", "columnas"):
        valor = esquema.get(eje)
        if isinstance(valor, dict):
            limites[eje] = {"minimo": valor.get("minimo", 1), "maximo": valor.get("maximo")}
        elif isinstance(valor, int):
            limites[eje] = {"minimo": 1, "maximo": None}
    return limites


def restricciones(corrida, esquema):
    """Restricciones de búsqueda: las de corrida.json mandan; si faltan, los rangos del esquema."""
    propias = dict(corrida.get("restricciones", {}))
    de_esquema = limites_esquema(esquema)
    for eje in ("filas", "columnas"):
        propias.setdefault(eje, de_esquema.get(eje, {"minimo": 1, "maximo": None}))
    propias.setdefault("tipos_pe", sorted(esquema.get("tipos_pe", {})))
    propias.setdefault("palabras_memoria_max", esquema.get("palabras_memoria"))
    propias.setdefault("anchos_vector_permitidos", [esquema["ancho_vector"]] if esquema.get("ancho_vector") else [])
    propias.setdefault("memoria_solo_en_bordes_oeste_este", False)
    propias.setdefault("requiere_memoria_y_computo", True)
    return propias


def cargar_contexto_conjunto(corrida, instancias):
    """contexto.json de una corrida con varias operaciones: el mismo cgra.json corre todas."""
    ops = list(instancias)
    primera = instancias[ops[0]]
    contexto = cargar_contexto({**corrida, "operacion": ops[0]}, primera)
    contexto.update({
        "operacion": "todas: " + ", ".join(ops),
        "operaciones": ops,
        "parametros_instancia": {op: i["parametros_instancia"] for op, i in instancias.items()},
        "datos_entrada": {op: relativa(i["memoria_inicial"]) for op, i in instancias.items()},
        "regiones_memoria": {op: i.get("regiones_memoria", {}) for op, i in instancias.items()},
        "regiones_salida": {op: i["regiones_salida"] for op, i in instancias.items()},
        "evaluacion_conjunto": "la configuración se evalúa con todas las operaciones (la CGRA no sabe cuál "
                               "corre); es correcta solo si lo es en todas y su costo es el promedio de los costos "
                               "de cada operación, con la latencia normalizada por la referencia en esa operación",
    })
    return contexto


def cargar_contexto(corrida, instancia):
    """Arma contexto.json a partir de la corrida y de la instancia ya preparada."""
    esquema = leer_json(resolver(corrida["esquema_cgra"]))
    tipos = esquema.get("tipos_pe", {})
    return {
        "programa_c": relativa(resolver(corrida["fuente_c"])),
        "operacion": corrida["operacion"],
        "parametros_instancia": instancia["parametros_instancia"],
        "datos_entrada": relativa(instancia["memoria_inicial"]),
        "resultado_esperado": "privado del evaluador (no se entrega al agente)",
        "regiones_memoria": instancia.get("regiones_memoria", {}),
        "regiones_salida": instancia["regiones_salida"],
        "grafo_dependencias": None,
        "restricciones": {
            **restricciones(corrida, esquema),
            "instrucciones_por_tipo": {letra: tipo["instrucciones"] for letra, tipo in tipos.items()},
            "nombres_tipo": {letra: tipo.get("nombre", letra) for letra, tipo in tipos.items()},
            "ancho_vector": esquema.get("ancho_vector"),
            "reglas_malla": relativa(SKILLS_01_09 / "mesh-compatibility.md"),
        },
        "parametros_evaluacion": {
            "w": corrida["pesos_area"],
            "K": corrida["K"],
            "formula_latencia": corrida["formula_latencia"],
            "tolerancias": corrida["tolerancias"],
            "penalizacion": corrida["penalizacion"],
            "configuracion_referencia": "corrida.json#/referencia",
            "limites": {**corrida["limites"], "tolerancia_empate_costo": corrida["tolerancia_empate_costo"]},
            "pendientes": corrida.get("pendientes", []),
        },
        "entrega": ENTREGA,
        "evaluador": "Skills 01–09 (generan los PE*.txt) + simulador Python (cgra/src/run_cgra.py)",
        "guias": {
            "instrucciones_agente": relativa(INSTRUCCIONES_AGENTE),
            "skills_01_09": relativa(SKILLS_01_09),
        },
    }
