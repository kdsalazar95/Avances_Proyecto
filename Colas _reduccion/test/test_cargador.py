"""Pruebas del arranque de la CGRA (memoria en blanco) y del cargador,
que copia la imagen de memoria y los programas antes de ejecutar."""

import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from cargador import cargar_imagen, cargar_programa, enlazar
from cgra import CGRA
from formato_cgra import INSTRUCCIONES_ISA, FormatoCGRA
from memoria_binaria import escribir_memoria
from programas_pe import ejecutar_cgra

DATOS_2X2 = os.path.join(_raiz_proyecto, "test", "datos", "reduccion_2x2")


def cgra_generica(filas, columnas, palabras_memoria):
    """CGRA con un solo tipo de PE que ejecuta todo el ISA: estas pruebas
    son del cargador, no de los tipos de PE (ver test_formato_cgra.py)."""
    tipos = {"G": {"nombre": "generico", "descripcion": "", "instrucciones": list(INSTRUCCIONES_ISA)}}
    return CGRA(FormatoCGRA(filas, columnas, palabras_memoria, tipos, [["G"] * columnas for _ in range(filas)]))


class TestCargador(unittest.TestCase):
    """Encender la CGRA y cargarle un programa, como hace el sistema operativo."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()
        self.ruta_memoria = os.path.join(self.directorio.name, "memoria.bin")
        # Imagen de la reducción 2x2: 12 + 12 + 12 + 1 = 37 palabras.
        escribir_memoria(
            self.ruta_memoria,
            {
                "a": [1, 2, 3, 4] * 3,
                "b": [1, 2, 3, 4] * 3,
                "c": [0] * 12,
                "result": [0],
            },
            {"a": (1, 12), "b": (1, 12), "c": (1, 12), "result": (1, 1)},
        )

    def tearDown(self):
        self.directorio.cleanup()

    def test_1_la_cgra_arranca_en_blanco(self):
        """Al encenderse, la CGRA debe tener toda la memoria en cero, sin regiones cargadas, y sus PEs sin programa y con los registros en cero."""
        cgra = cgra_generica(2, 2, 64)

        self.assertEqual(cgra.memoria, [0] * 64)
        self.assertEqual(cgra.simbolos, {})
        for pe_id in cgra.pe_ids():
            self.assertEqual(cgra.pe(pe_id)._instructions, [])
            self.assertEqual(cgra.pe(pe_id)._register, [0] * 16)

    def test_2_copia_la_imagen_y_el_resto_queda_en_cero(self):
        """El cargador debe copiar cada región a su dirección dentro de una memoria más grande, y el resto de la memoria debe seguir en cero."""
        cgra = cgra_generica(2, 2, 64)
        cargar_imagen(cgra, self.ruta_memoria)

        self.assertEqual(cgra.simbolos["a"], (0, 1, 12))
        self.assertEqual(cgra.simbolos["result"], (36, 1, 1))
        self.assertEqual(cgra.memoria[0:4], [1, 2, 3, 4])
        self.assertEqual(cgra.memoria[37:], [0] * (64 - 37))

    def test_3_la_imagen_se_puede_cargar_en_otra_direccion(self):
        """Con una dirección base distinta de 0, las regiones deben quedar desplazadas y la memoria anterior a la base en cero."""
        cgra = cgra_generica(2, 2, 64)
        cargar_imagen(cgra, self.ruta_memoria, base=10)

        self.assertEqual(cgra.simbolos["a"], (10, 1, 12))
        self.assertEqual(cgra.memoria[:10], [0] * 10)
        self.assertEqual(cgra.region("a"), [1, 2, 3, 4] * 3)

    def test_4_rechaza_una_imagen_que_no_cabe(self):
        """Si la imagen es más grande que la memoria de la CGRA, el cargador debe fallar en vez de cortarla."""
        cgra = cgra_generica(2, 2, 30)
        with self.assertRaisesRegex(ValueError, "ocupa 37 palabras"):
            cargar_imagen(cgra, self.ruta_memoria)

    def test_5_enlaza_ld_st_a_direcciones(self):
        """El cargador debe convertir cada banco[índice] de LD/ST en la dirección de esa palabra, y rechazar índices fuera de la región o regiones que no existen."""
        simbolos = {"a": (0, 1, 12), "result": (36, 1, 1)}
        programas = {"PE00": {0: {"op": "ld", "dst": "rA", "bank": "a", "index": 3},
                              1: {"op": "st", "bank": "result", "index": 0, "src": "acc"}}}
        enlazar(programas, simbolos)
        self.assertEqual(programas["PE00"][0]["addr"], 3)
        self.assertEqual(programas["PE00"][1]["addr"], 36)

        with self.assertRaisesRegex(ValueError, r"a\[12\] se sale"):
            enlazar({"PE00": {0: {"op": "ld", "dst": "rA", "bank": "a", "index": 12}}}, simbolos)
        with self.assertRaisesRegex(ValueError, "'z' no está"):
            enlazar({"PE00": {0: {"op": "ld", "dst": "rA", "bank": "z", "index": 0}}}, simbolos)

    def test_6_cada_pe_recibe_su_programa_y_se_ejecuta(self):
        """Después de cargar, cada PE debe tener su programa (24 instrucciones), y al ejecutar la reducción 2x2 debe quedar result = 60 en la memoria de la CGRA."""
        cgra = cgra_generica(2, 2, 64)
        programas = cargar_programa(cgra, self.ruta_memoria, DATOS_2X2)
        for pe_id in cgra.pe_ids():
            self.assertEqual(len(cgra.pe(pe_id)._instructions), 24)

        ejecutar_cgra(cgra.memoria, programas, 2, 2, malla=cgra.malla)

        self.assertEqual(cgra.region("c"), [2, 4, 6, 8] * 3)
        self.assertEqual(cgra.region("result"), [60])
        self.assertEqual(cgra.memoria[37:], [0] * (64 - 37))


if __name__ == "__main__":
    unittest.main()
