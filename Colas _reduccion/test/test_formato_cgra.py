"""Pruebas del formato de la CGRA (cgra.json) y de los tipos de PE."""

import json
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from cgra import CGRA
from formato_cgra import cargar_formato, validar_tipos
from programas_pe import descomponer_pe_id

MEMORIA = {"NOP", "MOV", "LD", "ST", "SEND", "RECV"}
COMPUTO = {"NOP", "MOV", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV"}


class TestFormatoCGRA(unittest.TestCase):
    """cgra.json describe la malla y qué puede hacer cada PE."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directorio.cleanup()

    def escribir_json(self, **cambios):
        datos = {
            "filas": 2,
            "columnas": 3,
            "palabras_memoria": 16,
            "tipos_pe": {
                "M": {"nombre": "memoria", "instrucciones": sorted(MEMORIA)},
                "C": {"nombre": "computo", "instrucciones": sorted(COMPUTO)},
            },
            "malla": ["M C M", "M C M"],
        }
        datos.update(cambios)
        ruta = os.path.join(self.directorio.name, "cgra.json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)
        return ruta

    def test_1_cgra_json_tiene_memoria_en_los_costados(self):
        """En src/cgra.json, los PEs de las columnas oeste y este deben ser de memoria (LD, ST, SEND, RECV, sin cálculo) y los del centro de cómputo (sin LD/ST)."""
        formato = cargar_formato()
        for pe_id in formato.pe_ids():
            _, columna = descomponer_pe_id(pe_id)
            if columna in (0, formato.columnas - 1):
                self.assertEqual(formato.nombre_tipo(pe_id), "memoria", pe_id)
                self.assertEqual(formato.instrucciones(pe_id), {op.lower() for op in MEMORIA})
            else:
                self.assertEqual(formato.nombre_tipo(pe_id), "computo", pe_id)
                self.assertEqual(formato.instrucciones(pe_id), {op.lower() for op in COMPUTO})

    def test_2_la_cgra_le_da_su_tipo_a_cada_pe(self):
        """Al encender la CGRA, cada PE debe quedar con el tipo y las instrucciones que dice el formato, y la memoria con el tamaño del formato."""
        cgra = CGRA(cargar_formato(self.escribir_json()))

        self.assertEqual(cgra.pe("PE00").tipo, "memoria")
        self.assertEqual(cgra.pe("PE01").tipo, "computo")
        self.assertEqual(cgra.pe("PE12").tipo, "memoria")
        self.assertEqual(len(cgra.memoria), 16)

    def test_3_un_pe_de_computo_no_puede_acceder_a_memoria(self):
        """Si un PE de cómputo intenta ejecutar LD o ST, el PE debe fallar: su hardware no tiene acceso a la memoria."""
        cgra = CGRA(cargar_formato(self.escribir_json()))
        pe = cgra.pe("PE01")

        with self.assertRaisesRegex(RuntimeError, "computo y no puede ejecutar LD"):
            pe.execute({"op": "ld", "dst": "rA", "addr": 0}, cgra.memoria)
        with self.assertRaisesRegex(RuntimeError, "computo y no puede ejecutar ST"):
            pe.execute({"op": "st", "src": "rA", "addr": 0}, cgra.memoria)

    def test_4_un_pe_de_memoria_no_puede_calcular(self):
        """Si un PE de memoria intenta ejecutar ADD o MUL, el PE debe fallar; sí puede hacer LD, SEND y RECV."""
        cgra = CGRA(cargar_formato(self.escribir_json()))
        cgra.memoria[0] = 7
        memoria_pe, computo_pe = cgra.pe("PE00"), cgra.pe("PE01")

        with self.assertRaisesRegex(RuntimeError, "memoria y no puede ejecutar ADD"):
            memoria_pe.execute({"op": "add", "dst": "rC", "src1": "rA", "src2": "rB"})
        memoria_pe.execute({"op": "ld", "dst": "rA", "addr": 0}, cgra.memoria)
        memoria_pe.execute({"op": "send", "dir": "east", "src": "rA"})
        computo_pe.execute({"op": "recv", "dir": "west", "dst": "rA"})
        self.assertEqual(computo_pe._register[0], 7)

    def test_5_validar_tipos_indica_pe_y_ciclo(self):
        """validar_tipos debe rechazar un programa que no respeta los tipos e indicar qué PE, de qué tipo, en qué ciclo y qué instrucción."""
        formato = cargar_formato(self.escribir_json())
        programas = {pe_id: {0: {"op": "nop"}} for pe_id in formato.pe_ids()}
        validar_tipos(programas, formato)

        programas["PE11"][0] = {"op": "ld", "dst": "rA", "bank": "a", "index": 0}
        with self.assertRaisesRegex(ValueError, r"PE11 \(computo\), ciclo 0: LD no está permitido"):
            validar_tipos(programas, formato)

    def test_6_rechaza_formatos_inconsistentes(self):
        """El formato debe fallar si la malla no tiene filas x columnas PEs, si usa un tipo no definido o si un tipo usa una instrucción que no existe."""
        with self.assertRaisesRegex(ValueError, "fila 1 de 'malla' tiene 2 PEs"):
            cargar_formato(self.escribir_json(malla=["M C M", "M C"]))
        with self.assertRaisesRegex(ValueError, "tipo 'X' no está definido"):
            cargar_formato(self.escribir_json(malla=["M C M", "M X M"]))
        with self.assertRaisesRegex(ValueError, "no existen: JMP"):
            cargar_formato(self.escribir_json(tipos_pe={"M": {"instrucciones": ["NOP", "JMP"]}}, malla=["M M M", "M M M"]))

    def test_7_la_malla_se_arma_sola_al_cambiar_el_tamano(self):
        """Con 'distribucion' (memoria en oeste y este), cambiar solo filas y columnas a 8x8 debe dar 8 filas 'M C C C C C C M', sin escribir la malla."""
        ruta = self.escribir_json(filas=8, columnas=8, distribucion={"oeste": "M", "este": "M", "resto": "C"})
        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        del datos["malla"]
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)

        formato = cargar_formato(ruta)

        self.assertEqual(formato.mapa(), ["M C C C C C C M"] * 8)
        self.assertEqual(len(formato.pe_ids()), 64)

    def test_8_una_malla_explicita_reemplaza_la_regla(self):
        """Si el JSON trae 'malla', esa distribución especial debe usarse en vez de la regla 'distribucion'."""
        ruta = self.escribir_json(distribucion={"oeste": "M", "resto": "C"}, malla=["C M C", "C M C"])

        self.assertEqual(cargar_formato(ruta).mapa(), ["C M C", "C M C"])

    def test_9_rechaza_una_regla_invalida(self):
        """La regla debe fallar si usa un borde que no existe o si no dice el tipo del resto de los PEs."""
        with open(self.escribir_json(), encoding="utf-8") as archivo:
            base = json.load(archivo)
        del base["malla"]
        for distribucion, mensaje in [
            ({"izquierda": "M", "resto": "C"}, "claves desconocidas: izquierda"),
            ({"oeste": "M"}, "necesita 'resto'"),
        ]:
            ruta = os.path.join(self.directorio.name, "regla.json")
            with open(ruta, "w", encoding="utf-8") as archivo:
                json.dump({**base, "distribucion": distribucion}, archivo)
            with self.assertRaisesRegex(ValueError, mensaje):
                cargar_formato(ruta)


if __name__ == "__main__":
    unittest.main()
