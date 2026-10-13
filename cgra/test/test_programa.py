"""Pruebas de los programas de los PEs (PE*.txt): lectura, validación,
ejecución en la CGRA y reporte de ciclos."""

import copy
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cargador import Cargador
from cgra import CGRA
from datos_prueba import escribir_reduccion_2x2
from formato_cgra import FormatoCGRA
from isa import ISA
from malla import Malla
from programa import Programa
from reporte_ciclos import ReporteCiclos


class ConReduccion2x2(unittest.TestCase):
    """Escribe la reducción 2x2 de datos_prueba.py en una carpeta temporal."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()
        self.ruta_memoria, self.ruta_instrucciones = escribir_reduccion_2x2(self.directorio.name)
        self.malla = Malla(2, 2)

    def tearDown(self):
        self.directorio.cleanup()


class TestEjecucionCGRA(ConReduccion2x2):
    """Lectura, validación y ejecución de los PE*.txt."""

    def test_1_carga_los_cuatro_programas_con_ciclos_alineados(self):
        """Los cuatro archivos deben cargar y cubrir los mismos 24 ciclos."""
        programa = Programa.leer(self.ruta_instrucciones, self.malla)

        self.assertEqual(set(programa.instrucciones), {"PE00", "PE01", "PE10", "PE11"})
        self.assertEqual(programa.ciclos(), list(range(24)))

    def test_2_ejecuta_mapa_y_reduccion_hasta_resultado_final(self):
        """La CGRA debe calcular c[i] = a[i] + b[i] y reducir los 12 valores a 60."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))
        cgra.ejecutar(Cargador(cgra).cargar(self.ruta_memoria, self.ruta_instrucciones))

        self.assertEqual(cgra.memoria.region("c"), [2, 4, 6, 8] * 3)
        self.assertEqual(cgra.memoria.region("result"), [60])
        self.assertEqual(cgra.malla.pe("PE00").leer_registro("acc"), 60)

    def test_3_rechaza_un_recv_sin_send_correspondiente(self):
        """La validación debe detectar si un RECV no tiene el SEND de su vecino en el mismo ciclo."""
        programa = Programa.leer(self.ruta_instrucciones, self.malla)
        programa["PE01"][19] = {"op": "nop"}

        with self.assertRaisesRegex(ValueError, "RECV east"):
            programa.validar(self.malla)

    def test_4_rechaza_una_instruccion_que_no_existe(self):
        """Una instrucción fuera del ISA debe fallar al leer el archivo, indicando el ciclo."""
        with open(os.path.join(self.ruta_instrucciones, "PE11.txt"), "a", encoding="utf-8") as archivo:
            archivo.write("24: XOR rA, rB, rC\n")

        with self.assertRaisesRegex(ValueError, "ciclo 24: sintaxis no válida"):
            Programa.leer(self.ruta_instrucciones, self.malla)


class TestEstadisticasCiclos(ConReduccion2x2):
    """Pruebas del reporte de ciclos de cómputo vs comunicación."""

    def setUp(self):
        super().setUp()
        self.programa = Programa.leer(self.ruta_instrucciones, self.malla)

    def sin_comunicacion(self):
        """Una copia del programa con los SEND/RECV cambiados por NOP."""
        programa = copy.deepcopy(self.programa)
        for pe_id in programa.instrucciones:
            for ciclo, instruccion in programa[pe_id].items():
                if instruccion["op"] in ISA.COMUNICACION:
                    programa[pe_id][ciclo] = {"op": "nop"}
        return programa

    def test_1_clasifica_los_24_ciclos_del_ejemplo(self):
        """De los 24 ciclos del ejemplo, solo los 2 con SEND/RECV deben contar como comunicación; el resto es cómputo y ninguno queda inactivo."""
        estadisticas = ReporteCiclos(self.programa, self.malla).contar()

        self.assertEqual(estadisticas, {"total": 24, "comunicacion": 2, "computo": 22, "inactivo": 0})

    def test_2_un_ciclo_totalmente_nop_cuenta_como_inactivo(self):
        """Si todos los PEs hacen NOP en un ciclo, ese ciclo debe contar como inactivo y no como cómputo."""
        for pe_id in self.programa.instrucciones:
            self.programa[pe_id][0] = {"op": "nop"}

        estadisticas = ReporteCiclos(self.programa, self.malla).contar()

        self.assertEqual(estadisticas["inactivo"], 1)
        self.assertEqual(estadisticas["computo"], 21)

    def test_3_guardar_escribe_los_totales_en_un_archivo(self):
        """El reporte guardado debe tener el total de ciclos y el desglose de cómputo/comunicación."""
        ruta = os.path.join(self.directorio.name, "reporte.txt")
        ReporteCiclos(self.programa, self.malla).guardar(ruta)

        with open(ruta, encoding="utf-8") as archivo:
            contenido = archivo.read()
        self.assertIn("Total de ciclos:        24", contenido)
        self.assertIn("Ciclos de comunicación: 2 (8.3%)", contenido)

    def test_4_lista_los_pasos_de_comunicacion_con_direccion_y_receptor(self):
        """El reporte debe listar cada SEND con su ciclo, emisor, receptor y dirección: PE01->PE00 y PE11->PE10 por el oeste, luego PE10->PE00 por el norte."""
        texto = "\n".join(ReporteCiclos(self.programa, self.malla).lineas())

        self.assertIn("Ciclo  19: PE01 -> PE00 (oeste)", texto)
        self.assertIn("Ciclo  19: PE11 -> PE10 (oeste)", texto)
        self.assertIn("Ciclo  21: PE10 -> PE00 (norte)", texto)

    def test_5_sin_comunicacion_lo_dice_explicitamente(self):
        """Si un programa no manda datos entre PEs, el reporte debe decirlo en vez de mostrar una lista vacía."""
        texto = "\n".join(ReporteCiclos(self.sin_comunicacion(), self.malla).lineas())

        self.assertIn("no manda datos entre PEs", texto)


if __name__ == "__main__":
    unittest.main()
