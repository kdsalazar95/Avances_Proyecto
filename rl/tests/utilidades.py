"""Utilidades de prueba: corrida mínima, instancia compartida, configuraciones y compiladores de prueba."""

import copy
import json
import re
import tempfile
from pathlib import Path

from rl.comun import FUENTE_OPERACIONES
from rl.evaluador.compilador import CompiladorReduccion
from rl.evaluador.entradas import preparar_instancia
from rl.evaluador.compilador_sustituto import compilar
from rl.herramientas.referencia import escribir_cgra, malla_filas

CORRIDA_BASE = {
    "id": "prueba",
    "operacion": "reduccion",
    "fuente_c": "cgra/operaciones/operaciones.c",
    "esquema_cgra": "cgra/src/cgra.json",
    "pesos_area": {"w1_MS": 4, "w2_MV": 8, "w3_CS": 1, "w4_CV": 2},
    "K": {"K_A": 1.0, "K_L": 1.0, "K_R": 1.0},
    "formula_latencia": "ciclos",
    "tolerancias": {"eps_abs": 1e-6, "eps_rel": 1e-6, "epsilon_cr": 1e-9},
    "penalizacion": {"P_fallo": 10.0},
    "tolerancia_empate_costo": 1e-9,
    "limites": {"max_iteraciones": 10, "paciencia": 3, "t_max_simulacion_s": 60, "t_max_total_s": None},
    "referencia": {"generar": {"malla": ["M E M", "M V M", "M E M"]}},
    "compilador": {"tipo": "reduccion_mcm", "N_corr": 1},
    "restricciones": {"memoria_solo_en_bordes_oeste_este": True, "requiere_memoria_y_computo": True},
}


def corrida(**cambios):
    datos = copy.deepcopy(CORRIDA_BASE)
    datos.update(cambios)
    return datos


class Temporal:
    """Directorio temporal con una instancia de reducción."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.instancia = preparar_instancia(FUENTE_OPERACIONES, "reduccion", self.raiz / "instancia")
        self.contador = 0

    def config(self, computo=("E", "V", "E"), con_este=True, nombre=None):
        """Carpeta de candidata con solo cgra.json + candidata.json (la acción del agente).

        ``computo``: tipo del PE de cómputo de cada fila; ``("E", "V", "E")`` es la referencia 3x3.
        """
        self.contador += 1
        destino = self.raiz / "acciones" / (nombre or f"a{self.contador}")
        escribir_cgra(destino, malla_filas(list(computo), con_este))
        return destino

    def programas(self, ruta_cgra, nombre=None):
        """PE*.txt generados por el sustituto de los skills para ``ruta_cgra``."""
        self.contador += 1
        destino = self.raiz / "programas" / (nombre or f"p{self.contador}")
        compilar(ruta_cgra, "reduccion", self.instancia["parametros_instancia"]["TAMANO_VECTOR"], destino)
        return destino

    def carpeta(self, nombre):
        self.contador += 1
        return self.raiz / f"{nombre}-{self.contador}"

    def cerrar(self):
        self._tmp.cleanup()


class CompiladorMutado(CompiladorReduccion):
    """Compila con el sustituto y luego aplica ``mutar(dir_salida, intento)`` (simula skills que se equivocan)."""

    def __init__(self, mutar):
        self.mutar = mutar
        self.intentos = []

    def compilar(self, ruta_cgra, instancia, dir_salida, errores_previos=None):
        self.intentos.append(errores_previos)
        resultado = super().compilar(ruta_cgra, instancia, dir_salida, errores_previos)
        if resultado["ok"]:
            self.mutar(Path(dir_salida), len(self.intentos))
        return resultado


class CompiladorCaido:
    """Simula que el LLM de los skills no está disponible."""

    def compilar(self, ruta_cgra, instancia, dir_salida, errores_previos=None):
        return {"ok": False, "infraestructura": True, "detalle": "no se encontró el LLM"}


def instruccion(ruta_pe, ciclo, nueva):
    """Reemplaza la instrucción de un ciclo en un PE*.txt (conserva el número)."""
    ruta = Path(ruta_pe)
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    for i, linea in enumerate(lineas):
        m = re.match(r"^(\d+):\s*(.*)$", linea)
        if m and int(m.group(1)) == ciclo:
            lineas[i] = f"{m.group(1)}: {nueva}"
            ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
            return
    raise ValueError(f"{ruta} no tiene el ciclo {ciclo}")


def buscar(ruta_pe, texto):
    """Ciclo de la primera instrucción que contiene ``texto``."""
    for linea in Path(ruta_pe).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\d+):\s*(.*)$", linea)
        if m and texto in m.group(2):
            return int(m.group(1))
    raise ValueError(f"{ruta_pe}: no aparece '{texto}'")


def editar_cgra(carpeta, cambio):
    ruta = Path(carpeta) / "cgra.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    cambio(datos)
    ruta.write_text(json.dumps(datos), encoding="utf-8")
