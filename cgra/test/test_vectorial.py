"""Pruebas de los PEs vectoriales: registros v0..v7 de ancho_vector palabras,
VLD/VST de palabras seguidas, operaciones carril a carril y VSEND/VRECV
del vector completo en un ciclo."""

import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from cargador import Cargador
from cgra import CGRA
from formato_cgra import FormatoCGRA, generar_malla
from isa import ISA
from malla import PE, Malla
from memoria import ImagenMemoria
from programa import Programa

# Tipos M/C con sus instrucciones vectoriales: M carga y envía, C calcula.
TIPOS_VECTORIALES = {
    "M": {"nombre": "memoria", "instrucciones": [
        "NOP", "MOV", "LD", "ST", "SEND", "RECV", "VLD", "VST", "VSEND", "VRECV",
    ]},
    "C": {"nombre": "computo", "instrucciones": [
        "NOP", "MOV", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV",
        "VADD", "VSUB", "VMUL", "VDIV", "VREDSUM", "VBCAST", "VSEND", "VRECV",
    ]},
}

# c = a + b y result = suma de c, con vectores de 4: PE00 (M) carga y guarda,
# PE01 (C) suma. Dos bloques de 4 elementos.
PROGRAMA_SUMA = {
    "PE00": [
        "VLD v0, a[0]", "VSEND east, v0", "VLD v1, b[0]", "VSEND east, v1", "NOP",
        "VRECV east, v2", "VST c[0], v2",
        "VLD v0, a[4]", "VSEND east, v0", "VLD v1, b[4]", "VSEND east, v1", "NOP",
        "VRECV east, v2", "VST c[4], v2",
        "NOP", "RECV east, rC", "ST result[0], rC",
    ],
    "PE01": [
        "MOV acc, 0.0", "VRECV west, v0", "NOP", "VRECV west, v1", "VADD v2, v0, v1",
        "VSEND west, v2", "VREDSUM rC, v2",
        "ADD acc, acc, rC", "VRECV west, v0", "NOP", "VRECV west, v1", "VADD v2, v0, v1",
        "VSEND west, v2", "VREDSUM rC, v2",
        "ADD acc, acc, rC", "SEND west, acc", "NOP",
    ],
}


class TestPEVectorial(unittest.TestCase):
    """Un PE con ancho_vector tiene registros vectoriales y ejecuta las instrucciones V..."""

    def setUp(self):
        self.pe = PE("PE00", ancho_vector=4)

    def test_1_parsea_y_reescribe_las_instrucciones_vectoriales(self):
        """Cada instrucción vectorial debe leerse del PE*.txt y volver a escribirse igual."""
        for texto in ("VLD v0, a[4]", "VST c[8], v2", "VADD v2, v0, v1", "VSUB v2, v0, v1",
                      "VMUL v2, v0, v1", "VDIV v2, v0, v1", "VREDSUM rC, v2", "VBCAST v1, rA",
                      "VSEND east, v0", "VRECV west, v3"):
            self.assertEqual(ISA.texto(ISA.parsear(texto)), texto)

    def test_2_opera_carril_a_carril_reduce_y_replica(self):
        """VADD y VMUL deben operar los 4 carriles a la vez, VREDSUM sumarlos en un escalar y VBCAST copiar un escalar en los 4."""
        self.pe.escribir_vector("v0", [1, 2, 3, 4])
        self.pe.escribir_vector("v1", [10, 20, 30, 40])
        self.pe.ejecutar(ISA.parsear("VADD v2, v0, v1"))
        self.pe.ejecutar(ISA.parsear("VMUL v3, v0, v1"))
        self.pe.ejecutar(ISA.parsear("VREDSUM rC, v2"))
        self.pe.escribir_registro("rA", 7)
        self.pe.ejecutar(ISA.parsear("VBCAST v4, rA"))

        self.assertEqual(self.pe.leer_vector("v2"), [11, 22, 33, 44])
        self.assertEqual(self.pe.leer_vector("v3"), [10, 40, 90, 160])
        self.assertEqual(self.pe.leer_registro("rC"), 110)
        self.assertEqual(self.pe.leer_vector("v4"), [7, 7, 7, 7])

    def test_3_vld_y_vst_usan_palabras_seguidas(self):
        """VLD debe leer 4 palabras seguidas desde su dirección y VST escribirlas, sin pasarse del final de la memoria."""
        memoria = list(range(10))
        self.pe.ejecutar({"op": "vld", "dst": "v0", "bank": "m", "index": 2, "addr": 2}, memoria)
        self.pe.ejecutar({"op": "vst", "bank": "m", "index": 6, "src": "v0", "addr": 6}, memoria)

        self.assertEqual(self.pe.leer_vector("v0"), [2, 3, 4, 5])
        self.assertEqual(memoria, [0, 1, 2, 3, 4, 5, 2, 3, 4, 5])
        with self.assertRaises(IndexError):
            self.pe.ejecutar({"op": "vld", "dst": "v0", "bank": "m", "index": 7, "addr": 7}, memoria)

    def test_4_vsend_y_vrecv_mueven_el_vector_completo(self):
        """Un VSEND debe dejar el vector entero en la cola del vecino en un solo envío; un RECV escalar no debe poder recibirlo."""
        malla = Malla(1, 2, ancho_vector=4)
        emisor, receptor = malla.pe("PE00"), malla.pe("PE01")
        emisor.escribir_vector("v0", [1, 2, 3, 4])

        emisor.ejecutar(ISA.parsear("VSEND east, v0"))
        receptor.ejecutar(ISA.parsear("VRECV west, v5"))
        self.assertEqual(receptor.leer_vector("v5"), [1, 2, 3, 4])

        emisor.ejecutar(ISA.parsear("VSEND east, v0"))
        with self.assertRaisesRegex(RuntimeError, "se recibe con VRECV"):
            receptor.ejecutar(ISA.parsear("RECV west, rA"))

    def test_5_sin_ancho_vector_no_hay_registros_vectoriales(self):
        """Un PE de una CGRA sin ancho_vector debe rechazar las instrucciones vectoriales."""
        pe = PE("PE00")
        with self.assertRaisesRegex(RuntimeError, "no tiene 'ancho_vector'"):
            pe.ejecutar(ISA.parsear("VBCAST v0, rA"))


class TestCGRAVectorial(unittest.TestCase):
    """cgra.json declara ancho_vector y los tipos M/C vectoriales; el cargador valida y enlaza los V..."""

    def test_1_tipos_vectoriales_necesitan_ancho_vector(self):
        """Un tipo con instrucciones vectoriales sin ancho_vector debe rechazarse; con ancho_vector, la leyenda debe mostrarlo."""
        with self.assertRaisesRegex(ValueError, "falta 'ancho_vector'"):
            FormatoCGRA(1, 2, 16, TIPOS_VECTORIALES, [["M", "C"]])

        formato = FormatoCGRA(1, 2, 16, TIPOS_VECTORIALES, [["M", "C"]], ancho_vector=4)
        self.assertIn("Vectores de 4 palabras (v0..v7)", formato.leyenda())
        self.assertIn("vld", formato.instrucciones("PE00"))
        self.assertNotIn("vadd", formato.instrucciones("PE00"))

    def test_2_vsend_se_empareja_solo_con_vrecv(self):
        """Un VSEND debe tener en el mismo ciclo un VRECV del vecino; un RECV escalar no sirve de pareja."""
        malla = Malla(1, 2, ancho_vector=4)
        Programa({"PE00": {0: ISA.parsear("VSEND east, v0")},
                  "PE01": {0: ISA.parsear("VRECV west, v0")}}).validar(malla)
        with self.assertRaisesRegex(ValueError, "VSEND east no coincide con PE01 VRECV west"):
            Programa({"PE00": {0: ISA.parsear("VSEND east, v0")},
                      "PE01": {0: ISA.parsear("RECV west, rA")}}).validar(malla)

    def test_3_un_vld_debe_caber_entero_en_su_region(self):
        """Al enlazar, las 4 palabras de un VLD deben estar dentro de la región; si no, el mensaje debe mostrar el rango."""
        simbolos = {"a": (0, 1, 8)}
        Programa({"PE00": {0: ISA.parsear("VLD v0, a[4]")}}).enlazar(simbolos, ancho_vector=4)
        with self.assertRaisesRegex(ValueError, r"a\[5\.\.8\] se sale de la región \(8 palabras\)"):
            Programa({"PE00": {0: ISA.parsear("VLD v0, a[5]")}}).enlazar(simbolos, ancho_vector=4)

    def test_4_filas_alternas_escalares_y_vectoriales(self):
        """Con 'filas_pares' y 'filas_impares' en 'distribucion', las filas pares deben ser M/C y las impares D/V, sea cual sea el tamaño; mezclarlas con otras claves debe fallar."""
        distribucion = {
            "filas_pares": {"oeste": "M", "este": "M", "resto": "C"},
            "filas_impares": {"oeste": "D", "este": "D", "resto": "V"},
        }
        self.assertEqual(generar_malla(4, 4, distribucion), [list("MCCM"), list("DVVD"), list("MCCM"), list("DVVD")])
        self.assertEqual(generar_malla(1, 3, distribucion), [list("MCM")])
        with self.assertRaisesRegex(ValueError, "mezcla"):
            generar_malla(2, 4, {**distribucion, "resto": "C"})

    def test_5_suma_vectorial_de_punta_a_punta(self):
        """Con un M y un C vectoriales (vectores de 4), el programa debe calcular c = a + b y result = 396, en 17 ciclos."""
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        b = [10, 20, 30, 40, 50, 60, 70, 80]
        with tempfile.TemporaryDirectory() as directorio:
            ruta_memoria = os.path.join(directorio, "memoria.bin")
            ImagenMemoria.desde_regiones(
                {"a": a, "b": b, "c": [0] * 8, "result": [0]},
                {"a": (1, 8), "b": (1, 8), "c": (1, 8), "result": (1, 1)},
            ).guardar(ruta_memoria)
            for pe_id, instrucciones in PROGRAMA_SUMA.items():
                with open(os.path.join(directorio, f"{pe_id}.txt"), "w", encoding="utf-8") as archivo:
                    for ciclo, instruccion in enumerate(instrucciones):
                        archivo.write(f"{ciclo:02d}: {instruccion}\n")

            cgra = CGRA(FormatoCGRA(1, 2, 64, TIPOS_VECTORIALES, [["M", "C"]], ancho_vector=4))
            programa = Cargador(cgra).cargar(ruta_memoria, directorio)
            cgra.ejecutar(programa)

        self.assertEqual(len(programa.ciclos()), 17)
        self.assertEqual(cgra.memoria.region("c"), [x + y for x, y in zip(a, b)])
        self.assertEqual(cgra.memoria.region("result"), [396])


if __name__ == "__main__":
    unittest.main()
