"""Pruebas de extremo a extremo: corre cada programa compilado por los
skills (reducción, matmul, convolución) en la CGRA de src/cgra.json,
respetando los tipos de PE, y compara la
memoria final con lo que calcula el .c del programa.

Sirven como red de seguridad para cambiar el simulador: si la CGRA
ejecuta bien las instrucciones, el resultado debe ser el mismo que el
del .c, sin importar cómo se ejecuten por dentro.

Los pe_instructions/ y memoria.bin de cada programa los generan los
skills y no están en el repositorio; si faltan, la prueba se omite.
"""

import copy
import os
import sys
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_raiz_proyecto, "src")
sys.path.insert(0, _src)

from cargador import cargar_programa
from cgra import CGRA
from formato_cgra import cargar_formato
from programas_pe import ejecutar_cgra


def referencia_reduccion(memoria, formas):
    """reduccion.c: c[i] = a[i] + b[i]; result = suma de c."""
    c = [x + y for x, y in zip(memoria["a"], memoria["b"])]
    return {"c": c, "result": [sum(c)]}


def referencia_matmul(memoria, formas):
    """matmul.c: result[i][j] = suma_k a[i][k] * b[k][j]."""
    n = formas["a"][0]
    a, b = memoria["a"], memoria["b"]
    return {"result": [
        sum(a[i * n + k] * b[k * n + j] for k in range(n))
        for i in range(n) for j in range(n)
    ]}


def referencia_convolucion(memoria, formas):
    """convolucion.c: result[i][j] = suma_ki,kj imagen[i+ki][j+kj] * kernel[ki][kj]."""
    n = formas["imagen"][1]
    k = formas["kernel"][0]
    salida = formas["result"][0]
    imagen, kernel = memoria["imagen"], memoria["kernel"]
    return {"result": [
        sum(imagen[(i + ki) * n + j + kj] * kernel[ki * k + kj] for ki in range(k) for kj in range(k))
        for i in range(salida) for j in range(salida)
    ]}


class TestProgramasCompilados(unittest.TestCase):
    """La CGRA debe dar el mismo resultado que el .c en cada programa."""

    # Carpeta con una subcarpeta por programa y formato de la CGRA a usar.
    carpeta_programas = _src
    ruta_formato = None  # src/cgra.json

    def correr_y_comparar(self, programa, referencia):
        carpeta = os.path.join(self.carpeta_programas, programa)
        ruta_memoria = os.path.join(carpeta, "memoria.bin")
        ruta_instrucciones = os.path.join(carpeta, "pe_instructions")
        if not (os.path.isfile(ruta_memoria) and os.path.isdir(ruta_instrucciones)):
            self.skipTest(f"{programa}: falta memoria.bin o pe_instructions/ (se generan con los skills)")

        # Igual que run_cgra.py: encender, cargar y ejecutar.
        formato = cargar_formato(self.ruta_formato) if self.ruta_formato else cargar_formato()
        cgra = CGRA(formato)  # con los tipos de PE del formato
        programas = cargar_programa(cgra, ruta_memoria, ruta_instrucciones)
        inicial, formas = cgra.regiones()
        esperado = referencia(copy.deepcopy(inicial), formas)
        ejecutar_cgra(cgra.memoria, programas, cgra.filas, cgra.columnas, malla=cgra.malla)
        memoria, _ = cgra.regiones()

        for banco, valores in esperado.items():
            self.assertEqual(memoria[banco], valores, f"{programa}: el banco '{banco}' no coincide con el .c")

    def test_1_reduccion(self):
        """Reducción: la CGRA debe dejar c = a + b y en result la suma de c, igual que reduccion.c."""
        self.correr_y_comparar("reduccion", referencia_reduccion)

    def test_2_matmul(self):
        """Matmul: la CGRA debe dejar en result el producto a x b, igual que matmul.c."""
        self.correr_y_comparar("matmul", referencia_matmul)

    def test_3_convolucion(self):
        """Convolución: la CGRA debe dejar en result la convolución de imagen con kernel, igual que convolucion.c."""
        self.correr_y_comparar("convolucion", referencia_convolucion)


class TestProgramas2x2(TestProgramasCompilados):
    """Los mismos tres programas, generados por los skills para una CGRA 2x2
    (test/datos/cgra_2x2/cgra.json):

        M C      con 2 columnas, memoria en oeste y este dejaría todos los PEs
        M C      de memoria; aquí la memoria va en el oeste y el cómputo al este.

    Las pruebas son las de TestProgramasCompilados, con otra CGRA y otros
    PE*.txt: el resultado tiene que ser el mismo que el del .c.
    """

    carpeta_programas = os.path.join(_raiz_proyecto, "test", "datos", "cgra_2x2")
    ruta_formato = os.path.join(carpeta_programas, "cgra.json")


if __name__ == "__main__":
    unittest.main()
