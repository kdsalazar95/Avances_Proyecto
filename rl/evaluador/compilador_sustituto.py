"""Sustituto determinista de los Skills 01–09 (E02) para pruebas y corridas sin LLM.

Se usa con ``"compilador": {"tipo": "reduccion_mcm"}``. Solo compila la
reducción en mallas de filas ``M X M`` o ``M X``, donde ``X`` es el PE de
cómputo de cada fila: E (escalar) o V (vectorial).
Con otra configuración falla, como fallarían unos skills que no saben mapearla.
La corrida real usa los Skills 01–09.

Schedule del sustituto:

1. Inicio: las filas escalares ponen ``acc`` en 0; las vectoriales ``v7`` en 0.
2. Trabajo, cada fila con su M oeste (en paralelo entre filas):
   - fila E: elemento i en 7 ciclos (M: LD a, SEND, LD b, SEND; X: ADD; X -> M; M: ST c);
   - fila V: bloque de ``ancho_vector`` elementos en 7 ciclos con VLD/VSEND/VADD/VST.
   Los bloques se reparten con un criterio voraz (a la fila que termina antes); los
   elementos que no llenan un bloque van a filas escalares (E).
3. Cierre: V pasa su vector acumulado a escalar con VREDSUM.
4. Reducción: cada fila envía su parcial, relevado fila por fila, a la fila
   escalar más alta (E), que lo suma. Si no hay filas escalares, los
   parciales viajan como vectores (VSEND) a la fila vectorial más alta.
5. Esa fila envía el total a su M oeste, que guarda ``result``.
"""

import json
from pathlib import Path

from ..comun import ARCHIVO_MANIFEST, CARPETA_PROGRAMAS

COMPUTO = ("E", "V")
ESCALARES = ("E",)
VECTORIALES = ("V",)
NECESARIAS = {
    "M": {"LD", "ST", "SEND", "RECV"},
    "E": {"MOV", "ADD", "SEND", "RECV"},
    "V": {"MOV", "VADD", "VBCAST", "VREDSUM", "SEND", "RECV", "VSEND", "VRECV"},
}
NECESARIAS_M_VECTOR = {"VLD", "VST", "VSEND", "VRECV"}
CICLOS_UNIDAD = 7


class NoSoportado(ValueError):
    """La configuración no es una malla que este compilador sepa programar."""


# --- análisis de la configuración ------------------------------------------

def _forma(cgra):
    """Devuelve ``(tipos de cómputo por fila, con_este, ancho_vector)`` o NoSoportado."""
    malla = cgra.get("malla")
    if not malla:
        raise NoSoportado("este compilador necesita una 'malla' explícita")
    filas = [fila.split() if isinstance(fila, str) else list(fila) for fila in malla]
    ancho = len(filas[0])
    if ancho not in (2, 3) or any(len(f) != ancho for f in filas):
        raise NoSoportado(f"solo compila filas 'M X M' o 'M X'; la malla es {malla}")
    for fila in filas:
        if fila[0] != "M" or (ancho == 3 and fila[2] != "M") or fila[1] not in COMPUTO:
            raise NoSoportado(f"solo compila filas 'M X M' o 'M X' con X en E o V; la malla es {malla}")
    computo = [fila[1] for fila in filas]
    tipos = cgra.get("tipos_pe", {})
    for letra in set(computo) | {"M"}:
        necesarias = NECESARIAS[letra] | (NECESARIAS_M_VECTOR if letra == "M" and set(computo) & set(VECTORIALES)
                                           else set())
        faltan = necesarias - set(tipos.get(letra, {}).get("instrucciones", []))
        if faltan:
            raise NoSoportado(f"el tipo {letra} no tiene {sorted(faltan)}")
    w = cgra.get("ancho_vector")
    if set(computo) & set(VECTORIALES) and not w:
        raise NoSoportado("hay PEs vectoriales y falta 'ancho_vector'")
    return computo, ancho == 3, w


def repartir(computo, n_elementos, w):
    """Reparte el trabajo: ``[[("escalar", i) | ("vector", j), ...] por fila]``.

    Bloques de ``w`` elementos: a la fila que terminaría antes (vectorial: 7 ciclos,
    escalar: 7·w). Elementos sobrantes: a filas escalares, uno por uno.
    """
    trabajo = [[] for _ in computo]
    fin = [0] * len(computo)
    vectoriales = [f for f, x in enumerate(computo) if x in VECTORIALES]
    escalares = [f for f, x in enumerate(computo) if x in ESCALARES]
    bloques = n_elementos // w if (w and vectoriales) else 0
    for b in range(bloques):
        costo = {f: CICLOS_UNIDAD if computo[f] in VECTORIALES else CICLOS_UNIDAD * w for f in range(len(computo))}
        f = min(range(len(computo)), key=lambda f: (fin[f] + costo[f], f))
        fin[f] += costo[f]
        if computo[f] in VECTORIALES:
            trabajo[f].append(("vector", b * w))
        else:
            trabajo[f] += [("escalar", i) for i in range(b * w, b * w + w)]
    resto = range(bloques * (w or 0) if vectoriales else 0, n_elementos)
    if len(resto) and not escalares:
        raise NoSoportado(f"{len(resto)} elementos no llenan un bloque de {w} y no hay PEs escalares (E)")
    for i in resto:
        f = min(escalares, key=lambda f: (fin[f] + CICLOS_UNIDAD, f))
        fin[f] += CICLOS_UNIDAD
        trabajo[f].append(("escalar", i))
    return trabajo


# --- generación de programas -----------------------------------------------

PASOS_ESCALAR = [
    ("LD rA, a[{i}]", "NOP"),
    ("SEND east, rA", "RECV west, rA"),
    ("LD rB, b[{i}]", "NOP"),
    ("SEND east, rB", "RECV west, rB"),
    ("NOP", "ADD rC, rA, rB"),
    ("RECV east, rC", "SEND west, rC"),
    ("ST c[{i}], rC", "ADD acc, acc, rC"),
]
PASOS_VECTOR = [
    ("VLD v0, a[{i}]", "NOP"),
    ("VSEND east, v0", "VRECV west, v0"),
    ("VLD v1, b[{i}]", "NOP"),
    ("VSEND east, v1", "VRECV west, v1"),
    ("NOP", "VADD v2, v0, v1"),
    ("VRECV east, v2", "VSEND west, v2"),
    ("VST c[{i}], v2", "VADD v7, v7, v2"),
]


def programas_reduccion(computo, n_elementos, con_este=True, w=4):
    """``({(fila, columna): [instrucciones]}, columnas, trabajo)``."""
    filas = len(computo)
    columnas = 3 if con_este else 2
    pes = {(f, c): [] for f in range(filas) for c in range(columnas)}

    def ciclo(por_pe):
        for pos in pes:
            pes[pos].append(por_pe.get(pos, "NOP"))

    trabajo = repartir(computo, n_elementos, w)
    modo_escalar = any(x in ESCALARES for x in computo)

    # 1. Inicio: E pone acc en 0; V pone v7 en 0 (MOV rT + VBCAST).
    ciclo({(f, 1): ("MOV acc, 0.0" if x == "E" else "MOV rT, 0.0") for f, x in enumerate(computo)})
    if any(x in VECTORIALES for x in computo):
        ciclo({(f, 1): "VBCAST v7, rT" for f, x in enumerate(computo) if x in VECTORIALES})

    # 2. Trabajo: cada fila con su M oeste; se rellena con NOP hasta la fila más larga.
    secuencias = {}
    for f, tareas in enumerate(trabajo):
        m, x = [], []
        for tipo, i in tareas:
            for paso_m, paso_x in (PASOS_VECTOR if tipo == "vector" else PASOS_ESCALAR):
                m.append(paso_m.format(i=i))
                x.append(paso_x.format(i=i))
        secuencias[f] = (m, x)
    largo = max((len(m) for m, _ in secuencias.values()), default=0)
    for k in range(largo):
        asignacion = {}
        for f, (m, x) in secuencias.items():
            if k < len(m):
                asignacion[(f, 0)] = m[k]
                asignacion[(f, 1)] = x[k]
        ciclo(asignacion)

    # 3. Cierre: el vector acumulado pasa a escalar (en modo escalar).
    destino_vector = None
    if modo_escalar:
        ciclo({(f, 1): "VREDSUM acc, v7" for f in range(filas) if computo[f] in VECTORIALES})
        destino = min(f for f, x in enumerate(computo) if x in ESCALARES)
    else:
        destino = destino_vector = 0

    # 4. Reducción: cada fila con trabajo envía su parcial al destino, relevado fila por fila.
    origenes = [f for f in range(filas) if f != destino and trabajo[f]]
    for f in origenes:
        paso = -1 if f > destino else 1
        direccion, opuesta = ("north", "south") if paso == -1 else ("south", "north")
        envio, recibo, reg = (("VSEND", "VRECV", "v6") if destino_vector is not None else ("SEND", "RECV", "rT"))
        origen_reg = "v7" if destino_vector is not None else "acc"
        actual = f
        while actual != destino:
            siguiente = actual + paso
            fuente = origen_reg if actual == f else reg
            ciclo({(actual, 1): f"{envio} {direccion}, {fuente}", (siguiente, 1): f"{recibo} {opuesta}, {reg}"})
            actual = siguiente
        if destino_vector is not None:
            ciclo({(destino, 1): "VADD v7, v7, v6"})
        else:
            ciclo({(destino, 1): "ADD acc, acc, rT"})
    if destino_vector is not None:
        ciclo({(destino, 1): "VREDSUM acc, v7"})

    # 5. El total va al M oeste del destino, que guarda result.
    ciclo({(destino, 1): "SEND west, acc", (destino, 0): "RECV east, rC"})
    ciclo({(destino, 0): "ST result[0], rC"})
    return pes, columnas, trabajo, destino


def compilar(ruta_cgra, operacion, n_elementos, salida):
    """Escribe pe_instructions/ y manifest.json en ``salida`` para la configuración dada."""
    if operacion != "reduccion":
        raise NoSoportado(f"el compilador sustituto solo compila 'reduccion', no '{operacion}'")
    cgra = json.loads(Path(ruta_cgra).read_text(encoding="utf-8"))
    computo, con_este, w = _forma(cgra)
    pes, columnas, trabajo, destino = programas_reduccion(computo, n_elementos, con_este, w)
    salida = Path(salida)
    carpeta_pe = salida / CARPETA_PROGRAMAS
    carpeta_pe.mkdir(parents=True, exist_ok=True)
    filas = len(computo)
    ancho = len(str(max(filas, columnas) - 1))
    ids = {pos: f"PE{pos[0]:0{ancho}d}{pos[1]:0{ancho}d}" for pos in pes}
    total = len(next(iter(pes.values())))
    digitos = max(2, len(str(total - 1)))
    for pos, instrucciones in sorted(pes.items()):
        letra = "M" if pos[1] != 1 else computo[pos[0]]
        tipo = cgra["tipos_pe"][letra]
        lineas = [
            f"// {ids[pos]} instruction list",
            f"// Mesh position: fila {pos[0]}, columna {pos[1]}",
            f"// PE type: {tipo.get('nombre', letra)} ({letra}) - {', '.join(tipo['instrucciones'])}",
            "// One instruction per execution cycle.",
            "",
        ] + [f"{c:0{digitos}d}: {inst}" for c, inst in enumerate(instrucciones)]
        (carpeta_pe / f"{ids[pos]}.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    manifest = {
        "operacion": "reduccion",
        "parametros_instancia": {"TAMANO_VECTOR": n_elementos},
        "filas": filas,
        "columnas": columnas,
        "malla": cgra["malla"],
        "ciclos": total,
        "output_pe": ids[(destino, 0)],
        "asignacion": {
            ids[(f, 1)]: {"tipo": computo[f], "servidor_memoria": ids[(f, 0)],
                          "escalares": [i for t, i in trabajo[f] if t == "escalar"],
                          "bloques_vectoriales": [i for t, i in trabajo[f] if t == "vector"]}
            for f in range(filas)
        },
        "generado_por": "rl/evaluador/compilador_sustituto.py (sustituto de los Skills 01–09)",
    }
    (salida / ARCHIVO_MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return salida
