"""Verificador independiente del bloque Dijkstra.

No usa nada del mapeador: revisa el ``ResultadoMapeo`` contra el grafo y
la malla antes de pasarlo al skill 08.

1. Cada operación está en un PE cuyo tipo tiene su instrucción.
2. No hay dos operaciones en el mismo ``(PE, ciclo)``.
3. Cada operación ocurre después de sus predecesores con tiempo para que
   llegue el dato: al menos ``1 + distancia Manhattan`` ciclos (cada salto
   cuesta un ciclo y el dato se usa al ciclo siguiente de recibirlo); los
   accesos a la misma posición de memoria, al menos 1 ciclo después.
4. Todas las operaciones del grafo están ubicadas.
5. El programa por PE, ejecutado de forma simbólica (qué valor hay en cada
   registro), hace exactamente eso: cada instrucción es la de su tipo de
   PE, cada SEND tiene su RECV (y cada VSEND su VRECV) en el vecino en el
   mismo ciclo, los escalares van en ``rA``.. y los vectores en ``v0``.., y
   cada operación lee en sus registros los valores de sus predecesores.
"""

from isa import ISA

DIRECCIONES = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
OPUESTA = {"north": "south", "south": "north", "west": "east", "east": "west"}
ARITMETICA = {
    "fadd": "add", "fsub": "sub", "fmul": "mul", "fdiv": "div",
    "vadd": "vadd", "vsub": "vsub", "vmul": "vmul", "vdiv": "vdiv",
}
# Campo de la instrucción parseada -> ¿registro vectorial?
CLASE_REGISTROS = {
    "send": {"src": False}, "recv": {"dst": False}, "vsend": {"src": True}, "vrecv": {"dst": True},
    "ld": {"dst": False}, "st": {"src": False}, "vld": {"dst": True}, "vst": {"src": True},
    "mov": {"dst": False}, "vbcast": {"dst": True, "src": False}, "vredsum": {"dst": False, "src": True},
    **{op: {"dst": False, "src1": False, "src2": False} for op in ("add", "sub", "mul", "div")},
    **{op: {"dst": True, "src1": True, "src2": True} for op in ("vadd", "vsub", "vmul", "vdiv")},
}


def verificar(grafo, malla, compatibilidad, resultado):
    """Lista de errores (vacía si el mapeo es correcto)."""
    if not resultado.valido:
        return [f"mapeo inválido: {resultado.motivo}"]
    errores = []
    pes = {pe.id: pe for pe in malla.pes}
    ubicacion = resultado.ubicacion

    # 4. Todas ubicadas.
    faltan = [op.id for op in grafo.operaciones if op.id not in ubicacion]
    if faltan:
        errores.append(f"operaciones sin ubicar: {faltan[:10]}")
        return errores

    # 1. Tipo compatible y 2. un solo uso por (PE, ciclo).
    usados = {}
    for op in grafo.operaciones:
        pe_id, ciclo = ubicacion[op.id]
        instruccion = compatibilidad.instruccion(op.op)
        if pe_id not in pes:
            errores.append(f"#{op.id} está en {pe_id}, que no existe")
            continue
        if instruccion not in pes[pe_id].instrucciones:
            errores.append(f"#{op.id} ({op.op}) en {pe_id} de tipo {pes[pe_id].letra}, que no tiene {instruccion}")
        if (pe_id, ciclo) in usados:
            errores.append(f"#{op.id} y #{usados[(pe_id, ciclo)]} están las dos en {pe_id}, ciclo {ciclo}")
        usados[(pe_id, ciclo)] = op.id

    # 3. Tiempo suficiente entre predecesores y sucesores.
    for op in grafo.operaciones:
        pe_id, ciclo = ubicacion[op.id]
        for entrada in op.entradas:
            otro_pe, otro_ciclo = ubicacion[entrada]
            distancia = abs(pes[pe_id].fila - pes[otro_pe].fila) + abs(pes[pe_id].columna - pes[otro_pe].columna)
            if ciclo < otro_ciclo + 1 + distancia:
                errores.append(
                    f"#{op.id} en ciclo {ciclo} no puede tener el dato de #{entrada} "
                    f"({otro_pe}, ciclo {otro_ciclo}, distancia {distancia})"
                )
        for previa in op.memoria:
            if ciclo <= ubicacion[previa][1]:
                errores.append(f"#{op.id} accede a memoria en el ciclo {ciclo}, antes que #{previa}")
    if errores:
        return errores
    return _ejecutar_simbolico(grafo, malla, resultado, pes)


def _ejecutar_simbolico(grafo, malla, resultado, pes):
    errores = []
    programa = resultado.programa
    if set(programa) != set(pes):
        return [f"el programa no tiene un PE*.txt por PE: {sorted(set(pes) ^ set(programa))}"]
    ciclos = {len(instrucciones) for instrucciones in programa.values()}
    if len(ciclos) != 1:
        return ["no todos los PEs tienen la misma cantidad de ciclos"]
    total = ciclos.pop()
    en_ciclo = {(pe_id, ciclo): op_id for op_id, (pe_id, ciclo) in resultado.ubicacion.items()}
    posicion = {(pe.fila, pe.columna): pe.id for pe in malla.pes}
    registros = {pe_id: {} for pe_id in pes}
    ejecutadas = set()

    for ciclo in range(total):
        parseadas = {}
        for pe_id, instrucciones in programa.items():
            origen = f"{pe_id}, ciclo {ciclo}"
            try:
                parseadas[pe_id] = ISA.parsear(instrucciones[ciclo], origen)
            except ValueError as error:
                errores.append(str(error))
                continue
            if parseadas[pe_id]["op"].upper() not in pes[pe_id].instrucciones:
                errores.append(f"{origen}: {instrucciones[ciclo]} no está permitido en el tipo {pes[pe_id].letra}")
            for campo, vectorial in CLASE_REGISTROS.get(parseadas[pe_id]["op"], {}).items():
                nombres = ISA.VECTORIALES if vectorial else ISA.REGISTROS
                if parseadas[pe_id][campo] not in nombres:
                    errores.append(f"{origen}: {instrucciones[ciclo]} usa {parseadas[pe_id][campo]}, "
                                   f"que no es un registro {'vectorial' if vectorial else 'escalar'}")
        # Primero los SEND, como el simulador.
        enviados = {}
        for pe_id, ins in parseadas.items():
            if ins["op"] in ("send", "vsend"):
                nombre, esperada = ins["op"].upper(), ISA.PAREJA[ins["op"]]
                fila, columna = pes[pe_id].fila, pes[pe_id].columna
                df, dc = DIRECCIONES[ins["dir"]]
                vecino = posicion.get((fila + df, columna + dc))
                if vecino is None:
                    errores.append(f"{pe_id}, ciclo {ciclo}: {nombre} {ins['dir']} sale de la malla")
                    continue
                pareja = parseadas.get(vecino)
                if not pareja or pareja["op"] != esperada or pareja["dir"] != OPUESTA[ins["dir"]]:
                    errores.append(f"{pe_id}, ciclo {ciclo}: {nombre} {ins['dir']} sin {esperada.upper()} en {vecino}")
                enviados[(vecino, OPUESTA[ins["dir"]])] = registros[pe_id].get(ins["src"])
        for pe_id, ins in parseadas.items():
            op_id = en_ciclo.get((pe_id, ciclo))
            origen = f"{pe_id}, ciclo {ciclo}"
            tipo = ins["op"]
            if tipo in ("send", "vsend", "nop", "recv", "vrecv"):
                if op_id is not None:
                    errores.append(f"{origen}: debería ejecutar #{op_id} y tiene {programa[pe_id][ciclo]}")
                if tipo in ("recv", "vrecv"):
                    if (pe_id, ins["dir"]) not in enviados:
                        errores.append(f"{origen}: {tipo.upper()} {ins['dir']} sin {ISA.PAREJA[tipo].upper()} del vecino")
                    registros[pe_id][ins["dst"]] = enviados.get((pe_id, ins["dir"]))
                continue
            if op_id is None:
                errores.append(f"{origen}: {programa[pe_id][ciclo]} no corresponde a ninguna operación")
                continue
            op = grafo.operacion(op_id)
            ejecutadas.add(op_id)
            esperado = None
            if op.op == "load" and tipo == "ld" and (ins["bank"], ins["index"]) == (op.banco, op.indice):
                registros[pe_id][ins["dst"]] = op_id
                esperado = True
            elif op.op == "const" and tipo == "mov" and ins["imm"] == float(op.valor):
                registros[pe_id][ins["dst"]] = op_id
                esperado = True
            elif op.op == "store" and tipo == "st" and (ins["bank"], ins["index"]) == (op.banco, op.indice):
                esperado = registros[pe_id].get(ins["src"]) == op.entradas[0]
            elif op.op == "vload" and tipo == "vld" and (ins["bank"], ins["index"]) == (op.banco, op.indice):
                registros[pe_id][ins["dst"]] = op_id
                esperado = True
            elif op.op == "vstore" and tipo == "vst" and (ins["bank"], ins["index"]) == (op.banco, op.indice):
                esperado = registros[pe_id].get(ins["src"]) == op.entradas[0]
            elif op.op in ("vbcast", "vredsum") and tipo == op.op:
                esperado = registros[pe_id].get(ins["src"]) == op.entradas[0]
                registros[pe_id][ins["dst"]] = op_id
            elif ARITMETICA.get(op.op) == tipo:
                leidos = (registros[pe_id].get(ins["src1"]), registros[pe_id].get(ins["src2"]))
                esperado = leidos == tuple(op.entradas)
                registros[pe_id][ins["dst"]] = op_id
            if not esperado:
                errores.append(f"{origen}: {programa[pe_id][ciclo]} no hace #{op_id} ({op.texto()}) con sus operandos")
        if len(errores) > 20:
            break
    faltan = set(resultado.ubicacion) - ejecutadas
    if faltan and not errores:
        errores.append(f"operaciones que el programa no ejecuta: {sorted(faltan)[:10]}")
    return errores
