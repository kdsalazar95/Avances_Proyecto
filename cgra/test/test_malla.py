"""Pruebas de la malla: conexiones entre PEs vecinos y comunicación por colas."""

import os
import sys
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from malla import Malla


class TestConectarMallaHorizontal(unittest.TestCase):
    """Cada PE queda conectado con su vecino de la derecha en ambos
    sentidos (este->oeste y oeste->este), con colas independientes."""

    def setUp(self):
        self.malla = Malla(2, 3)

    def fila(self, f):
        return [self.malla.pe(self.malla.id_de(f, c)) for c in range(self.malla.columnas)]

    def test_1_extremos_de_fila_sin_conexion_faltante(self):
        """El primer PE de cada fila no debe tener colas hacia el oeste, y el último no debe tener colas hacia el este (no hay vecino de ese lado)."""
        for f in range(self.malla.filas):
            fila = self.fila(f)
            self.assertIsNone(fila[0].entradas["west"])
            self.assertIsNone(fila[0].salidas["west"])
            self.assertIsNone(fila[-1].entradas["east"])
            self.assertIsNone(fila[-1].salidas["east"])

    def test_2_canal_hacia_el_este_es_compartido(self):
        """La cola de salida al este de un PE debe ser la misma que la cola de entrada desde el oeste de su vecino."""
        fila = self.fila(0)
        for c in range(len(fila) - 1):
            self.assertIs(fila[c].salidas["east"], fila[c + 1].entradas["west"])

    def test_3_canal_hacia_el_oeste_es_compartido(self):
        """La cola de salida al oeste de un PE debe ser la misma que la cola de entrada desde el este de su vecino izquierdo."""
        fila = self.fila(0)
        for c in range(len(fila) - 1):
            self.assertIs(fila[c + 1].salidas["west"], fila[c].entradas["east"])

    def test_4_los_dos_sentidos_usan_colas_distintas(self):
        """La cola este->oeste y la cola oeste->este entre el mismo par de vecinos no deben ser la misma (si no, los datos que van y los que vienen se mezclarían)."""
        fila = self.fila(0)
        self.assertIsNot(fila[0].salidas["east"], fila[0].entradas["east"])


class TestConectarMallaVertical(unittest.TestCase):
    """Cada PE queda conectado con su vecino de abajo en ambos sentidos
    (norte->sur y sur->norte), con colas independientes."""

    def setUp(self):
        self.malla = Malla(3, 2)

    def pe(self, f, c):
        return self.malla.pe(self.malla.id_de(f, c))

    def test_1_extremos_de_columna_sin_conexion_faltante(self):
        """El PE de arriba de cada columna no debe tener colas hacia el norte, y el de abajo no debe tener colas hacia el sur."""
        for c in range(self.malla.columnas):
            self.assertIsNone(self.pe(0, c).entradas["north"])
            self.assertIsNone(self.pe(0, c).salidas["north"])
            self.assertIsNone(self.pe(self.malla.filas - 1, c).entradas["south"])
            self.assertIsNone(self.pe(self.malla.filas - 1, c).salidas["south"])

    def test_2_canal_hacia_el_sur_es_compartido(self):
        """La cola de salida al sur de un PE debe ser la misma que la cola de entrada desde el norte del PE de abajo."""
        for f in range(self.malla.filas - 1):
            for c in range(self.malla.columnas):
                self.assertIs(self.pe(f, c).salidas["south"], self.pe(f + 1, c).entradas["north"])

    def test_3_canal_hacia_el_norte_es_compartido(self):
        """La cola de salida al norte de un PE debe ser la misma que la cola de entrada desde el sur del PE de arriba."""
        for f in range(self.malla.filas - 1):
            for c in range(self.malla.columnas):
                self.assertIs(self.pe(f + 1, c).salidas["north"], self.pe(f, c).entradas["south"])


class TestComunicacionBidireccional(unittest.TestCase):
    """Dos PEs vecinos pueden mandarse datos de ida y de vuelta."""

    def ida_y_vuelta(self, valor_ida, valor_a_sumar):
        """PE00 le manda ``valor_ida`` a PE01 por el este; PE01 le suma
        ``valor_a_sumar`` y lo regresa por el oeste. Devuelve lo que PE00
        recibió de vuelta en rB."""
        malla = Malla(1, 2)
        pe0, pe1 = malla.pe("PE00"), malla.pe("PE01")

        pe0.ejecutar({"op": "mov", "dst": "rA", "imm": valor_ida})
        pe0.ejecutar({"op": "send", "dir": "east", "src": "rA"})
        pe1.ejecutar({"op": "recv", "dir": "west", "dst": "rA"})
        pe1.ejecutar({"op": "mov", "dst": "rB", "imm": valor_a_sumar})
        pe1.ejecutar({"op": "add", "dst": "rC", "src1": "rA", "src2": "rB"})
        pe1.ejecutar({"op": "send", "dir": "west", "src": "rC"})
        pe0.ejecutar({"op": "recv", "dir": "east", "dst": "rB"})

        return pe0.leer_registro("rB")

    def test_1_envios_no_se_mezclan_entre_sentidos(self):
        """Un SEND hacia el este debe llegar a la cola de entrada del vecino y no a la cola por la que el PE recibe lo que viene del este."""
        malla = Malla(1, 2)
        pe0, pe1 = malla.pe("PE00"), malla.pe("PE01")

        pe0.escribir_registro("rA", 9)
        pe0.ejecutar({"op": "send", "dir": "east", "src": "rA"})

        self.assertEqual(len(pe1.entradas["west"].datos), 1)
        self.assertEqual(len(pe0.entradas["east"].datos), 0)

    def test_2_ida_y_vuelta_tres_veces(self):
        """La comunicación de ida y vuelta debe dar el resultado correcto con datos distintos, sin mezclarse con la anterior."""
        for i, (valor_ida, valor_a_sumar, esperado) in enumerate([(5, 100, 105), (0, 7, 7), (-2, 10, 8)]):
            with self.subTest(caso=i):
                self.assertEqual(self.ida_y_vuelta(valor_ida, valor_a_sumar), esperado)

    def test_3_send_sin_vecino_falla(self):
        """Un SEND hacia un lado donde no hay vecino debe fallar con un mensaje claro."""
        pe = Malla(1, 2).pe("PE00")
        with self.assertRaisesRegex(RuntimeError, "no tiene vecino hacia west"):
            pe.ejecutar({"op": "send", "dir": "west", "src": "rA"})


if __name__ == "__main__":
    unittest.main()
