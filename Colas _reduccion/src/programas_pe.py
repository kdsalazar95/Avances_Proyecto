"""Lógica para cargar, validar y ejecutar programas sobre una CGRA de tamaño filas x columnas."""

import re
from pathlib import Path

from formato_texto import fmt, imprimir_titulo
from pe_malla import crear_malla, conectar_malla


DIRECCIONES_OPUESTAS = {"north": "south", "south": "north", "east": "west", "west": "east"}
DESPLAZAMIENTOS = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
DIRECCIONES_EN_ESPANOL = {"north": "norte", "south": "sur", "west": "oeste", "east": "este"}

PATRON_CICLO = re.compile(r"^(\d+):\s*(.*)$")
# Valida instrucciones con el formato "ciclo: instrucción".
# (\d+) captura el número de ciclo y (.*) captura la instrucción.
# ciclo: instruccion
# Ejemplo: 0: ADD acc, acc, r0

PATRON_MEMORIA = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")
# Valida operandos de memoria con el formato "banco[índice]".
# ([A-Za-z_]\w*) captura el nombre del banco y (\d+) captura el índice.
#
# Ejemplo: a[0], b[5]

# Las funciones de utilidad para dar formato a números, imprimir títulos y
# cajas de texto (fmt, imprimir_titulo, caja, formatear_lista, formato_pe)
# viven en formato_texto.py.

#================== Lectura de instrucciones .txt ===========================
def numero(texto):
    """Lee una constante entera o flotante de MOV."""
    valor = float(texto)
    return int(valor) if valor.is_integer() else valor

# Esta funcion convierte los numeros que vienen como texto a numeros de Python

# Ejemplo: mov r0, 5
# numero("5") -> 5 (devuelve 5)
# lo mismo sucede si fuera un valor flotante: mov r0, 5.5

def operando_memoria(texto, ruta, ciclo):
    coincidencia = PATRON_MEMORIA.fullmatch(texto) # comprobacion del formato correcto
    if not coincidencia:
        raise ValueError(f"{ruta}, ciclo {ciclo}: memoria inválida '{texto}'")
    return coincidencia.group(1), int(coincidencia.group(2)) # separa a[3] en ("a", 3)

# La funcion interpreta una direccion de memoria (identifica que banco de memoria y que posicion
#se estan utilizando)

# a[0]                 a: banco,  3: indice
# b[5]
# result[0]

# Instrucciones como LD acc, a[3]
# se puede convertir en un diccionario para que el PE lo pueda ejecutar
# por ejemplo:
# {"op": "ld", "dst": "acc", "bank": "a", "index": 3}

def parsear_instruccion(texto, ruta, ciclo):
    """Convierte la sintaxis del skill 08 a un diccionario para ``PE.execute``."""
    partes = texto.replace(",", " ").split()
    if not partes:
        raise ValueError(f"{ruta}, ciclo {ciclo}: instrucción vacía")
    op = partes[0].upper()
    argumentos = partes[1:]

    if op == "NOP" and not argumentos:
        return {"op": "nop"}
    if op == "MOV" and len(argumentos) == 2:
        return {"op": "mov", "dst": argumentos[0], "imm": numero(argumentos[1])}
    if op == "LD" and len(argumentos) == 2:
        banco, indice = operando_memoria(argumentos[1], ruta, ciclo)
        return {"op": "ld", "dst": argumentos[0], "bank": banco, "index": indice}
    if op == "ST" and len(argumentos) == 2:
        banco, indice = operando_memoria(argumentos[0], ruta, ciclo)
        return {"op": "st", "bank": banco, "index": indice, "src": argumentos[1]}
    if op in {"ADD", "SUB", "MUL", "DIV"} and len(argumentos) == 3:
        return {"op": op.lower(), "dst": argumentos[0], "src1": argumentos[1], "src2": argumentos[2]}
    if op == "SEND" and len(argumentos) == 2:
        return {"op": "send", "dir": argumentos[0].lower(), "src": argumentos[1]}
    if op == "RECV" and len(argumentos) == 2:
        return {"op": "recv", "dir": argumentos[0].lower(), "dst": argumentos[1]}
    raise ValueError(f"{ruta}, ciclo {ciclo}: sintaxis no válida: '{texto}'")


# Toma una instruccion escrita en un .txt y traduce al formato que entienda el PE.execute()
# por ejemplo, la instruccion ADD acc, acc, r0 se traduce a:
# {"op": "add", "dst": "acc", "src1": "acc", "src2": "r0"}

def cargar_programa_pe(ruta):
    """Carga un archivo del skill 08 y devuelve ``{ciclo: instrucción}``."""
    programa = {}
    for numero_linea, linea in enumerate(Path(ruta).read_text(encoding="utf-8").splitlines(), start=1): # Abre el archivo y lo recorre linea por linea
        texto = linea.split("//", 1)[0].strip()
        if not texto:
            continue
        coincidencia = PATRON_CICLO.fullmatch(texto)
        if not coincidencia:
            raise ValueError(f"{ruta}, línea {numero_linea}: se esperaba 'NN: INSTRUCCIÓN'")
        ciclo = int(coincidencia.group(1))
        if ciclo in programa:
            raise ValueError(f"{ruta}, ciclo {ciclo}: ciclo duplicado")
        programa[ciclo] = parsear_instruccion(coincidencia.group(2), ruta, ciclo)
    if not programa:
        raise ValueError(f"{ruta}: programa vacío")
    return programa

# Carga y convierte el contenido de un archivo PExx.txt en un programa que la cgra puede ejecutar

def cargar_programas_pe(directorio, filas, columnas):
    """Carga y valida los PE{fila}{columna}.txt de una malla filas x columnas."""
    directorio = Path(directorio)
    programas = {}
    for pe_id in generar_pe_ids(filas, columnas):
        ruta = directorio / f"{pe_id}.txt"
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe {ruta}")
        programas[pe_id] = cargar_programa_pe(ruta)
    validar_programas(programas, filas, columnas)
    return programas

# Esta funcion trabaja con todos los PEs de la malla (filas x columnas):
# carga los .txt que representan los programas de los PEs y los valida antes de ejecutar la CGRA.

#================== Identificadores de PE ===================================

def ancho_id_pe(filas, columnas):
    """Cuántos dígitos ocupa cada coordenada en un ID de PE, según el
    índice más grande que puede aparecer en la malla.

    Con filas y columnas <= 10 da 1 (PE00, PE77, ...), igual que antes;
    con mallas más grandes (ej. 12x12) da 2 (PE0100 para fila=1,
    columna=0), para que un índice de dos dígitos no se confunda con
    dos índices de un dígito.
    """
    return len(str(max(filas, columnas) - 1))


def descomponer_pe_id(pe_id):
    """Separa un ID como ``PE0100`` en ``(fila, columna) = (1, 0)``.

    Ambas coordenadas ocupan el mismo ancho dentro del ID (ver
    ``generar_pe_ids``), así que alcanza con partir el número al medio.
    """
    numeros = pe_id[2:]
    ancho = len(numeros) // 2
    return int(numeros[:ancho]), int(numeros[ancho:])


def generar_pe_ids(filas, columnas):
    """Genera los identificadores PE de una malla de tamaño filas x columnas.

    Coincide con el nombre de archivo de cada PE (PE00.txt, PE01.txt, ...).
    Por ejemplo, con filas=2 y columnas=2 devuelve
    ("PE00", "PE01", "PE10", "PE11"), igual que antes.
    """
    ancho = ancho_id_pe(filas, columnas)
    return tuple(f"PE{f:0{ancho}d}{c:0{ancho}d}" for f in range(filas) for c in range(columnas))


def vecino(pe_id, direccion, filas, columnas):
    """Obtiene el PE vecino de un PE dado según la dirección indicada."""

    # Obtiene la fila y columna del PE
    fila, columna = descomponer_pe_id(pe_id)

    # Obtiene el desplazamiento correspondiente a la dirección
    delta_fila, delta_columna = DESPLAZAMIENTOS.get(direccion, (None, None))

    # Verifica que la dirección sea válida
    if delta_fila is None:
        raise ValueError(f"Dirección inválida: {direccion}")

    # Calcula la posición del PE vecino
    destino_fila, destino_columna = fila + delta_fila, columna + delta_columna

    # Verifica que el vecino esté dentro de la malla filas x columnas
    if not (0 <= destino_fila < filas and 0 <= destino_columna < columnas):
        raise ValueError(f"{pe_id}: no tiene vecino hacia {direccion}")

    # Devuelve el identificador del PE vecino, con el mismo ancho que los demás IDs
    ancho = ancho_id_pe(filas, columnas)
    return f"PE{destino_fila:0{ancho}d}{destino_columna:0{ancho}d}"


def validar_programas(programas, filas, columnas):
    """Aplica las comprobaciones estructurales relevantes del skill 09."""
    ciclos = set(next(iter(programas.values())))
    if ciclos != set(range(max(ciclos) + 1)):
        raise ValueError("Los ciclos deben comenzar en 0 y ser consecutivos")
    for pe_id, programa in programas.items():
        if set(programa) != ciclos:
            raise ValueError(f"{pe_id}: rango de ciclos distinto al resto")

    for ciclo in sorted(ciclos):
        for emisor, programa in programas.items():
            instruccion = programa[ciclo]
            if instruccion["op"] == "send":
                receptor = vecino(emisor, instruccion["dir"], filas, columnas)
                esperada = DIRECCIONES_OPUESTAS[instruccion["dir"]]
                recibida = programas[receptor][ciclo]
                if recibida["op"] != "recv" or recibida["dir"] != esperada:
                    raise ValueError(f"Ciclo {ciclo}: {emisor} SEND {instruccion['dir']} no coincide con {receptor} RECV {esperada}")
            elif instruccion["op"] == "recv":
                receptor = emisor
                emisor = vecino(receptor, instruccion["dir"], filas, columnas)
                esperada = DIRECCIONES_OPUESTAS[instruccion["dir"]]
                enviada = programas[emisor][ciclo]
                if enviada["op"] != "send" or enviada["dir"] != esperada:
                    raise ValueError(f"Ciclo {ciclo}: {receptor} RECV {instruccion['dir']} no coincide con {emisor} SEND {esperada}")
# verifica que:
# 1. Los ciclos comienzan en 0 y son consecutivos
#               ↓
# 2. Todos los PEs tienen los mismos ciclos
#               ↓
# 3. Los SEND y RECV coinciden correctamente
#               ↓
#         Programa válido

def texto_instruccion(instruccion):
    """Vuelve a escribir una instrucción ya parseada con la sintaxis del skill 08."""
    op = instruccion["op"]
    if op == "nop":
        return "NOP"
    if op == "mov":
        return f"MOV {instruccion['dst']}, {fmt(instruccion['imm'])}"
    if op == "ld":
        return f"LD {instruccion['dst']}, {instruccion['bank']}[{instruccion['index']}]"
    if op == "st":
        return f"ST {instruccion['bank']}[{instruccion['index']}], {instruccion['src']}"
    if op in {"add", "sub", "mul", "div"}:
        return f"{op.upper()} {instruccion['dst']}, {instruccion['src1']}, {instruccion['src2']}"
    if op == "send":
        return f"SEND {instruccion['dir']}, {instruccion['src']}"
    if op == "recv":
        return f"RECV {instruccion['dir']}, {instruccion['dst']}"
    return op.upper()


def efecto_instruccion(pe, instruccion, memoria):
    """Describe qué dejó la instrucción recién ejecutada: el registro o la
    posición de memoria que escribió, o el valor que mandó a la cola."""
    op = instruccion["op"]
    if op == "st":
        banco, indice = instruccion["bank"], instruccion["index"]
        if "addr" in instruccion:
            return f"{banco}[{indice}] = {fmt(memoria[instruccion['addr']])}  (dirección {instruccion['addr']})"
        return f"{banco}[{indice}] = {fmt(memoria[banco][indice])}"
    if op == "send":
        return f"envía {fmt(pe._register[pe._indice_registro(instruccion['src'])])}"
    if "dst" in instruccion:
        return f"{instruccion['dst']} = {fmt(pe._register[pe._indice_registro(instruccion['dst'])])}"
    return ""


def ejecutar_cgra(memoria, programas, filas, columnas, mostrar_traza=False, malla=None):
    """Ejecuta una instrucción por PE y por ciclo; SEND se procesa primero.

    La CGRA no sabe qué operación corre: solo ejecuta las instrucciones de
    cada PE. Si ``mostrar_traza`` es verdadero, imprime en cada ciclo qué
    ejecutó cada PE (sin los NOP) y qué dejó en registros o memoria.

    ``malla`` es la de una CGRA ya cargada (ver cargador.py); si no se da,
    se crea una malla nueva.
    """
    if malla is None:
        malla = crear_malla(filas, columnas)
        conectar_malla(malla)
    pe_ids = generar_pe_ids(filas, columnas)
    pes = {}
    for pe_id in pe_ids:
        fila, columna = descomponer_pe_id(pe_id)
        pes[pe_id] = malla[fila][columna]

    if mostrar_traza:
        imprimir_titulo("TRAZA: ciclo | PE | instrucción | efecto")
    for ciclo in sorted(next(iter(programas.values()))):
        instrucciones_del_ciclo = {pe_id: programas[pe_id][ciclo] for pe_id in pe_ids}
        efectos = {}

        # Primero todos los SEND, para que el RECV del vecino encuentre el
        # dato en la cola dentro del mismo ciclo.
        for pe_id, instruccion in instrucciones_del_ciclo.items():
            if instruccion["op"] == "send":
                pes[pe_id].execute(instruccion, memoria)
                efectos[pe_id] = efecto_instruccion(pes[pe_id], instruccion, memoria)
        for pe_id, instruccion in instrucciones_del_ciclo.items():
            if instruccion["op"] != "send":
                pes[pe_id].execute(instruccion, memoria)
                efectos[pe_id] = efecto_instruccion(pes[pe_id], instruccion, memoria)

        if mostrar_traza:
            for pe_id in pe_ids:
                instruccion = instrucciones_del_ciclo[pe_id]
                if instruccion["op"] != "nop":
                    print(f"  {ciclo:>5} | {pe_id} | {texto_instruccion(instruccion):<22} | {efectos[pe_id]}")
    return malla

# Flujo de la funcion:
# Crear malla filas x columnas
#        ↓
# Conectar PEs
#        ↓
# Ciclo 0
#  ├─ ejecutar SEND
#  └─ ejecutar demás instrucciones
#        ↓
# Ciclo 1
#  ├─ ejecutar SEND
#  └─ ejecutar demás instrucciones
#        ↓
# Ciclo 2
#  ├─ ejecutar SEND
#  └─ ejecutar demás instrucciones
#       ↓
#      ...
#       ↓
# Devolver CGRA ejecutada
