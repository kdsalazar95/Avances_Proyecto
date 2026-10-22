"""Pruebas del formato de la CGRA (cgra.json): tamaño de la malla, de la memoria y tipos de PE."""

import json
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from cgra import CGRA
from formato_cgra import FormatoCGRA, generar_malla, validar_tipos
from isa import ISA
from malla import Malla
from programa import Programa


TIPOS = {
    "M": {"nombre": "memoria", "instrucciones": ["NOP", "LD", "ST"]},
    "C": {"nombre": "computo", "instrucciones": ["NOP", "ADD"]},
}


class TestFormatoCGRA(unittest.TestCase):
    """cgra.json describe el tamaño de la malla, de la memoria de datos y el tipo de cada PE."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directorio.cleanup()

    def escribir_json(self, **cambios):
        datos = {"filas": 2, "columnas": 3, "palabras_memoria": 16}
        datos.update(cambios)
        ruta = os.path.join(self.directorio.name, "cgra.json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)
        return ruta

    def test_1_src_cgra_json_es_valido(self):
        """src/cgra.json, el que leen run_cgra.py y los skills, debe cargarse sin errores y dar una malla y una memoria con al menos un elemento."""
        formato = FormatoCGRA.cargar()
        self.assertGreater(formato.filas, 0)
        self.assertGreater(formato.columnas, 0)
        self.assertGreater(formato.palabras_memoria, 0)

    def test_2_la_cgra_toma_su_tamano_del_formato(self):
        """Al encender la CGRA, la malla debe tener filas x columnas PEs y la memoria el tamaño del formato."""
        cgra = CGRA(FormatoCGRA.cargar(self.escribir_json()))

        self.assertEqual((cgra.filas, cgra.columnas), (2, 3))
        self.assertEqual(len(cgra.malla.ids()), 6)
        self.assertEqual(cgra.malla.ids(), ("PE00", "PE01", "PE02", "PE10", "PE11", "PE12"))
        self.assertEqual(len(cgra.memoria), 16)

    def test_3_rechaza_un_formato_incompleto(self):
        """Si falta filas, columnas o palabras_memoria, el formato no se debe cargar, y el mensaje debe decir qué falta."""
        ruta = self.escribir_json()
        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        del datos["palabras_memoria"]
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)

        with self.assertRaisesRegex(ValueError, "faltan los campos palabras_memoria"):
            FormatoCGRA.cargar(ruta)

    def test_4_rechaza_tamanos_invalidos(self):
        """Una malla sin PEs, una memoria sin palabras o un tamaño que no es entero deben fallar al cargar el formato."""
        with self.assertRaisesRegex(ValueError, "al menos una fila"):
            FormatoCGRA.cargar(self.escribir_json(filas=0))
        with self.assertRaisesRegex(ValueError, "al menos una palabra"):
            FormatoCGRA.cargar(self.escribir_json(palabras_memoria=0))
        with self.assertRaisesRegex(ValueError, "columnas deben ser números enteros"):
            FormatoCGRA.cargar(self.escribir_json(columnas="4"))

    def test_5_rechaza_un_json_que_no_existe_o_esta_mal_escrito(self):
        """Un formato que no existe o que no es JSON válido debe fallar con un mensaje claro."""
        with self.assertRaisesRegex(ValueError, "No existe el formato"):
            FormatoCGRA.cargar(os.path.join(self.directorio.name, "no_existe.json"))

        ruta = os.path.join(self.directorio.name, "roto.json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write("{filas: 4")
        with self.assertRaisesRegex(ValueError, "JSON inválido"):
            FormatoCGRA.cargar(ruta)

    def test_6_asigna_tipo_e_instrucciones_a_cada_pe(self):
        """Con una malla de tipos, cada PE debe tener el nombre y las instrucciones de su letra."""
        formato = FormatoCGRA(3, 4, 16, TIPOS, [["M", "C", "C", "M"]] * 3)

        self.assertEqual(formato.nombre_tipo("PE00"), "memoria")
        self.assertEqual(formato.instrucciones("PE01"), {"nop", "add"})

    def test_7_rechaza_instrucciones_no_permitidas_por_tipo(self):
        """Si un PE usa una instrucción que su tipo no tiene, debe fallar al cargar (indicando el PE y la instrucción) y también si se intenta ejecutar."""
        formato = FormatoCGRA(1, 2, 16, TIPOS, [["M", "C"]])
        programa = Programa({"PE00": {0: {"op": "add"}}})

        with self.assertRaisesRegex(ValueError, "PE00.*ADD no está permitido"):
            validar_tipos(programa, formato)

        pe = CGRA(formato).malla.pe("PE00")
        self.assertEqual(pe.tipo, "memoria")
        with self.assertRaisesRegex(RuntimeError, "PE00 es de tipo memoria y no puede ejecutar ADD"):
            pe.ejecutar({"op": "add", "dst": "rC", "src1": "rA", "src2": "rB"})

    def test_8_rechaza_un_tipo_con_operacion_que_no_existe(self):
        """Un tipo de PE no puede declarar instrucciones que no están en el ISA."""
        with self.assertRaisesRegex(ValueError, "no existen: JMP"):
            FormatoCGRA(1, 1, 16, {"C": {"instrucciones": ["JMP"]}}, [["C"]])

    def test_9_distribucion_por_bordes_o_malla_explicita(self):
        """Con 'distribucion', las columnas oeste y este deben ser M y el resto C; si el JSON trae 'malla', se usa tal cual."""
        distribucion = {"oeste": "M", "este": "M", "resto": "C"}
        ruta = self.escribir_json(filas=3, columnas=4, tipos_pe=TIPOS, distribucion=distribucion)
        self.assertEqual(FormatoCGRA.cargar(ruta).mapa(), ["M C C M"] * 3)

        ruta = self.escribir_json(filas=2, columnas=2, tipos_pe=TIPOS, distribucion=distribucion, malla=["M C", "C M"])
        self.assertEqual(FormatoCGRA.cargar(ruta).mapa(), ["M C", "C M"])

    def test_10_ciclo_de_filas_escalar_vectorial_computo(self):
        """Con 'ciclo_filas' las filas deben repetir las reglas en orden (E, V, C, E...) con M en oeste y este, sea cual sea el tamaño; una lista vacía o mezclarla con otras claves debe fallar."""
        ciclo = {"ciclo_filas": [{"oeste": "M", "este": "M", "resto": letra} for letra in "EVC"]}
        self.assertEqual(generar_malla(6, 4, ciclo), [list(fila) for fila in ("MEEM", "MVVM", "MCCM") * 2])
        self.assertEqual(generar_malla(2, 3, ciclo), [list("MEM"), list("MVM")])
        with self.assertRaisesRegex(ValueError, "al menos una regla"):
            generar_malla(2, 3, {"ciclo_filas": []})
        with self.assertRaisesRegex(ValueError, "mezcla"):
            generar_malla(2, 3, {**ciclo, "resto": "C"})

    def test_11_cada_tipo_de_src_cgra_json_ejecuta_solo_lo_suyo(self):
        """En src/cgra.json, M solo debe acceder a memoria, E solo hacer aritmética escalar, V solo vectorial y C ambas; lo demás debe rechazarse."""
        formato = FormatoCGRA.cargar()
        letras = {}
        for pe in Malla(formato.filas, formato.columnas).ids():
            letras.setdefault(formato.letra(pe), pe)
        casos = {
            "M": (["LD rA, a[0]", "VLD v0, a[0]"], ["ADD rA, rA, rB", "VADD v0, v0, v1"]),
            "E": (["ADD rA, rA, rB"], ["VADD v0, v0, v1", "VSEND east, v0", "LD rA, a[0]"]),
            "V": (["VADD v0, v0, v1", "VREDSUM rC, v2", "SEND west, rC"], ["ADD rA, rA, rB", "LD rA, a[0]"]),
            "C": (["ADD rA, rA, rB", "VADD v0, v0, v1", "VREDSUM rC, v2"], ["VLD v0, a[0]"]),
        }
        self.assertEqual(set(letras), set(casos))
        for letra, (aceptadas, rechazadas) in casos.items():
            pe = letras[letra]
            for instruccion in aceptadas:
                validar_tipos(Programa({pe: {0: ISA.parsear(instruccion)}}), formato)
            for instruccion in rechazadas:
                with self.assertRaisesRegex(ValueError, "no está permitido"):
                    validar_tipos(Programa({pe: {0: ISA.parsear(instruccion)}}), formato)

if __name__ == "__main__":
    unittest.main()
