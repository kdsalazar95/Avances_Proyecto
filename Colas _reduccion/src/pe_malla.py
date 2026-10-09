"""Modela una malla (grid) de PEs (Processing Elements) que se
comunican entre sí a través de colas FIFO, de forma similar a un
arreglo sistólico o a la capa de interconexión de un CGRA.

Cada PE ejecuta su propio programa de instrucciones (aritméticas y de
comunicación) sobre sus propios registros, y puede enviar/recibir
datos con sus 4 vecinos (norte, sur, oeste, este) en ambos sentidos,
usando una cola distinta para cada sentido de cada dirección.
"""


class QUEUE:
    """Cola simple tipo FIFO (First In, First Out).

    Se usa para simular un canal de comunicación unidireccional entre
    dos PEs vecinos: el PE emisor la usa para "enviar" datos (push) y
    el PE receptor la usa para "recibir" (pop). Cada sentido de
    comunicación entre dos vecinos usa su propia QUEUE, para que los
    datos que van y los que vienen nunca se mezclen.
    """

    def __init__(self):
        """Crea una cola vacía."""
        self.datos = []

    def push(self, dato):
        """Agrega un dato al final de la cola."""
        self.datos.append(dato)

    def pop(self):
        """Saca y devuelve el dato más antiguo de la cola (FIFO)."""
        if not self.datos:
            raise RuntimeError("RECV intentó leer una cola FIFO vacía")
        return self.datos.pop(0)  # FIFO: sale el primero que entró


class PE:
    """Elemento de procesamiento (Processing Element) de la malla.

    Tiene sus propios registros e instrucciones, y hasta 8 colas de
    comunicación: dos por cada vecino (norte, sur, oeste, este) — una
    de salida (para enviarle datos a ese vecino) y otra de entrada
    (para recibir datos que ese vecino le mande). Así, dos PEs vecinos
    pueden mandarse datos el uno al otro en cualquier momento, sin que
    se mezclen en la misma cola.
    """

    def __init__(self, pe_id=None):
        """Crea un PE nuevo, con registros en cero y sin conexiones.

        pe_id: identificador opcional (por ejemplo, una tupla
        (fila, columna) que indica su posición en la malla).
        """
        self.id = pe_id
        # Tipo del PE (ver formato_cgra.py) y las instrucciones que su
        # hardware puede ejecutar. None = sin restricción (PE genérico).
        self.tipo = None
        self.instrucciones_permitidas = None
        self._register = [0] * 16
        self._pc = 0
        self._instructions = []

        # Colas hacia/desde el vecino norte
        self._queue_n_out = None  # para enviar datos hacia el norte
        self._queue_n_in = None   # para recibir datos que vienen del norte

        # Colas hacia/desde el vecino sur
        self._queue_s_out = None  # para enviar datos hacia el sur
        self._queue_s_in = None   # para recibir datos que vienen del sur

        # Colas hacia/desde el vecino oeste
        self._queue_w_out = None  # para enviar datos hacia el oeste
        self._queue_w_in = None   # para recibir datos que vienen del oeste

        # Colas hacia/desde el vecino este
        self._queue_e_out = None  # para enviar datos hacia el este
        self._queue_e_in = None   # para recibir datos que vienen del este

    # Nombres definidos por los skills para los registros del ISA textual.
    # Los enteros se siguen aceptando para no romper los programas y pruebas
    # existentes que usan registros 0..15 directamente.
    REGISTROS = {
        "rA": 0,
        "rB": 1,
        "rC": 2,
        "rT": 3,
        "acc": 4,
    }

    def _indice_registro(self, registro):
        """Convierte un nombre ISA (por ejemplo ``rA``) o un entero a índice."""
        if isinstance(registro, str):
            if registro not in self.REGISTROS:
                raise ValueError(f"Registro desconocido: {registro}")
            return self.REGISTROS[registro]
        if not isinstance(registro, int) or not 0 <= registro < len(self._register):
            raise ValueError(f"Índice de registro inválido: {registro}")
        return registro

    def load_instructions(self, instructions):
        """Carga un programa (lista de instrucciones) y reinicia el PC."""
        self._instructions = instructions
        self._pc = 0

    def step(self, memoria=None):
        """Ejecuta una sola instrucción y avanza el PC.

        Devuelve False si ya no quedan instrucciones por ejecutar,
        True si se ejecutó una instrucción.
        """
        if self._pc >= len(self._instructions):
            return False
        inst = self._instructions[self._pc]
        self.execute(inst, memoria)
        self._pc += 1
        return True

    def run(self, memoria=None):
        """Ejecuta todas las instrucciones restantes, una por una,
        hasta que no queden más."""
        while self.step(memoria):
            pass

    def execute(self, inst, memoria=None):
        """Ejecuta una única instrucción según su operación (`inst["op"]`):
        add/sub/mov operan sobre los registros locales; send_n/send_s/
        send_w/send_e mandan un dato al vecino correspondiente; recv_n/
        recv_s/recv_w/recv_e reciben un dato de ese vecino. La
        comunicación funciona en las 4 direcciones y en ambos sentidos.
        """
        op = inst["op"].lower()
        if self.instrucciones_permitidas is not None:
            # send_e/recv_w del formato antiguo cuentan como SEND/RECV.
            base = op.split("_")[0]
            if base not in self.instrucciones_permitidas:
                raise RuntimeError(f"PE{self.id} es de tipo {self.tipo} y no puede ejecutar {base.upper()}")

        # El formato antiguo usa regA/regB/regC. El parser del ISA textual
        # utiliza dst/src1/src2 para reflejar la sintaxis ADD dst, src1, src2.
        def reg(nombre_antiguo, nombre_nuevo):
            return self._indice_registro(inst.get(nombre_nuevo, inst.get(nombre_antiguo)))

        def banco_e_indice():
            if memoria is None:
                raise RuntimeError(f"{op.upper()} requiere memoria compartida")
            # Con la CGRA cargada, el cargador ya resolvió a[3] a una
            # dirección de la memoria plana (ver cargador.py).
            if "addr" in inst:
                direccion = inst["addr"]
                if not 0 <= direccion < len(memoria):
                    raise IndexError(f"Dirección fuera de la memoria: {direccion}")
                return None, direccion
            banco = inst["bank"]
            indice = inst["index"]
            if banco not in memoria:
                raise KeyError(f"Banco de memoria inexistente: {banco}")
            if not isinstance(indice, int) or not 0 <= indice < len(memoria[banco]):
                raise IndexError(f"Índice inválido: {banco}[{indice}]")
            return banco, indice

        if op == "add":
            self._register[reg("regC", "dst")] = self._register[reg("regA", "src1")] + self._register[reg("regB", "src2")]
        elif op == "sub":
            self._register[reg("regC", "dst")] = self._register[reg("regA", "src1")] - self._register[reg("regB", "src2")]
        elif op == "mul":
            self._register[reg("regC", "dst")] = self._register[reg("regA", "src1")] * self._register[reg("regB", "src2")]
        elif op == "div":
            self._register[reg("regC", "dst")] = self._register[reg("regA", "src1")] / self._register[reg("regB", "src2")]
        elif op == "mov":
            self._register[reg("regC", "dst")] = inst["imm"]
        elif op == "ld":
            banco, indice = banco_e_indice()
            celdas = memoria if banco is None else memoria[banco]
            self._register[reg(None, "dst")] = celdas[indice]
        elif op == "st":
            banco, indice = banco_e_indice()
            celdas = memoria if banco is None else memoria[banco]
            celdas[indice] = self._register[reg(None, "src")]
        elif op == "nop":
            pass
        # --- enviar datos a un vecino ---
        elif op in {"send_n", "send_s", "send_w", "send_e", "send"}:
            direccion = inst.get("dir", op[-1] if op != "send" else None)
            colas_salida = {
                "north": self._queue_n_out, "south": self._queue_s_out,
                "west": self._queue_w_out, "east": self._queue_e_out,
                "n": self._queue_n_out, "s": self._queue_s_out,
                "w": self._queue_w_out, "e": self._queue_e_out,
            }
            cola = colas_salida.get(direccion)
            if cola is None:
                raise RuntimeError(f"PE{self.id} no tiene vecino hacia {direccion}")
            self_reg = reg("regA", "src")
            cola.push(self._register[self_reg])
        # --- recibir datos de un vecino ---
        elif op in {"recv_n", "recv_s", "recv_w", "recv_e", "recv"}:
            direccion = inst.get("dir", op[-1] if op != "recv" else None)
            colas_entrada = {
                "north": self._queue_n_in, "south": self._queue_s_in,
                "west": self._queue_w_in, "east": self._queue_e_in,
                "n": self._queue_n_in, "s": self._queue_s_in,
                "w": self._queue_w_in, "e": self._queue_e_in,
            }
            cola = colas_entrada.get(direccion)
            if cola is None:
                raise RuntimeError(f"PE{self.id} no tiene vecino hacia {direccion}")
            self._register[reg("regA", "dst")] = cola.pop()
        else:
            raise ValueError(f"Operación desconocida: {op}")

    def __repr__(self):
        """Representación en texto del PE, útil para imprimirlo y
        depurar (muestra su id, el pc actual y sus registros)."""
        return f"PE(id={self.id}, pc={self._pc}, registers={self._register})"


def crear_malla(filas, columnas):
    """Crea una malla (grid) de PEs de tamaño filas x columnas.
    Devuelve una lista de listas: malla[fila][columna] -> PE
    """
    return [
        [PE(pe_id=(f, c)) for c in range(columnas)]
        for f in range(filas)
    ]


def conectar_malla_horizontal(malla):
    """Conecta cada PE con su vecino de la derecha en la misma fila,
    en AMBOS sentidos: se crean dos colas por cada par de vecinos, una
    para que el de la izquierda le mande datos al de la derecha
    (send_e / recv_w), y otra para que el de la derecha le mande datos
    al de la izquierda (send_w / recv_e).
    """
    for fila in malla:
        for c in range(len(fila) - 1):
            izquierda, derecha = fila[c], fila[c + 1]

            hacia_el_este = QUEUE()
            izquierda._queue_e_out = hacia_el_este
            derecha._queue_w_in = hacia_el_este

            hacia_el_oeste = QUEUE()
            derecha._queue_w_out = hacia_el_oeste
            izquierda._queue_e_in = hacia_el_oeste


def conectar_malla_vertical(malla):
    """Conecta cada PE con su vecino de abajo en la misma columna, en
    AMBOS sentidos: se crean dos colas por cada par de vecinos, una
    para que el de arriba le mande datos al de abajo (send_s / recv_n),
    y otra para que el de abajo le mande datos al de arriba
    (send_n / recv_s).
    """
    filas = len(malla)
    columnas = len(malla[0]) if filas > 0 else 0
    for f in range(filas - 1):
        for c in range(columnas):
            arriba, abajo = malla[f][c], malla[f + 1][c]

            hacia_el_sur = QUEUE()
            arriba._queue_s_out = hacia_el_sur
            abajo._queue_n_in = hacia_el_sur

            hacia_el_norte = QUEUE()
            abajo._queue_n_out = hacia_el_norte
            arriba._queue_s_in = hacia_el_norte


def conectar_malla(malla):
    """Conecta la malla completa de una sola vez: llama a
    conectar_malla_horizontal() y conectar_malla_vertical(), para que
    cada PE quede comunicado en ambos sentidos con todos sus vecinos
    existentes (norte/sur/este/oeste).
    """
    conectar_malla_horizontal(malla)
    conectar_malla_vertical(malla)
