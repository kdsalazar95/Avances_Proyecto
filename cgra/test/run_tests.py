"""
Corre todas las pruebas de test/, pero muestra los resultados de
forma más fácil de leer: agrupados por sección, uno por línea, con un
check (✔) o una X según haya pasado o fallado, y usando la explicación
en español de cada prueba en vez de solo su nombre técnico.

Cómo correrlo:
    uv run python3 test/run_tests.py
"""
import io
import os
import sys
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_malla
import test_programa
import test_resultados
import test_formato_cgra
import test_cargador
import test_operaciones
import test_simulador


# Nombres de sección más amigables para cada clase de prueba.
NOMBRES_DE_SECCION = {
    "TestConectarMallaHorizontal": "Conexión horizontal (este \u2194 oeste)",
    "TestConectarMallaVertical": "Conexión vertical (norte \u2194 sur)",
    "TestComunicacionBidireccional": "Comunicación entre dos PEs (ida y vuelta)",
    "TestEjecucionCGRA": "Ejecución de programas de la CGRA",
    "TestEstadisticasCiclos": "Ciclos de cómputo vs comunicación",
    "TestMemoriaBinaria": "Memoria binaria de la CGRA (memoria.bin)",
    "TestPresentacionResultado": "Presentación del resultado desde la memoria",
    "TestFormatoCGRA": "Formato de la CGRA (cgra.json)",
    "TestCargador": "Arranque de la CGRA y cargador (memoria en blanco)",
    "TestOperaciones": "Operaciones (operaciones.c genera memoria.bin)",
    "TestSimulador": "Simulador completo (una operación o todas)",
}


class ResultadoAmigable(unittest.TextTestResult):
    """Versión de TestResult que, en vez de imprimir una sola línea
    con puntos o nombres técnicos, imprime cada prueba en su propia
    línea, agrupada por sección, con un check o una X y la
    descripción en español (el docstring de la prueba)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seccion_actual = None
        self.exitosas = 0
        self.fallidas = []

    def _descripcion(self, test):
        # Junta todas las líneas del docstring en una sola oración
        # fluida (en vez de cortar solo la primera línea a la mitad).
        doc = test._testMethodDoc
        if doc:
            lineas = [l.strip() for l in doc.strip().splitlines()]
            return " ".join(lineas)
        return test._testMethodName

    def _seccion(self, test):
        nombre_clase = type(test).__name__
        return NOMBRES_DE_SECCION.get(nombre_clase, nombre_clase)

    def startTest(self, test):
        super().startTest(test)
        seccion = self._seccion(test)
        if seccion != self.seccion_actual:
            if self.seccion_actual is not None:
                print()
            print(f"{seccion}")
            print("-" * len(seccion))
            self.seccion_actual = seccion

    def addSuccess(self, test):
        super().addSuccess(test)
        self.exitosas += 1
        print(f"  \u2714  {self._descripcion(test)}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.fallidas.append(test)
        print(f"  \u2716  {self._descripcion(test)}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"  -  {self._descripcion(test)}  (omitida: {reason})")

    def addError(self, test, err):
        super().addError(test, err)
        self.fallidas.append(test)
        print(f"  \u2716  {self._descripcion(test)}  (error inesperado)")


def main():
    cargador = unittest.TestLoader()

    # Se arma la suite en un orden narrativo específico (primero
    # horizontal, luego vertical, luego comunicación bidireccional),
    # en vez del orden alfabético por defecto de unittest.
    orden_de_clases = [
        test_malla.TestConectarMallaHorizontal,
        test_malla.TestConectarMallaVertical,
        test_malla.TestComunicacionBidireccional,
        test_programa.TestEjecucionCGRA,
        test_programa.TestEstadisticasCiclos,
        test_resultados.TestMemoriaBinaria,
        test_resultados.TestPresentacionResultado,
        test_formato_cgra.TestFormatoCGRA,
        test_cargador.TestCargador,
        test_operaciones.TestOperaciones,
        test_simulador.TestSimulador,
    ]
    suite = unittest.TestSuite()
    for clase in orden_de_clases:
        suite.addTests(cargador.loadTestsFromTestCase(clase))

    # La salida técnica por defecto de unittest ("Ran 9 tests...", "OK")
    # se manda a un buffer aparte y se descarta: nuestro propio resumen
    # de abajo la reemplaza con algo más fácil de leer.
    buffer_silenciado = io.StringIO()
    runner = unittest.TextTestRunner(resultclass=ResultadoAmigable, verbosity=0, stream=buffer_silenciado)
    resultado = runner.run(suite)

    total = resultado.testsRun
    exitosas = total - len(resultado.failures) - len(resultado.errors)

    print()
    print("=" * 40)
    if exitosas == total:
        print(f"Todo bien: {exitosas}/{total} pruebas pasaron \u2714")
    else:
        print(f"{exitosas}/{total} pruebas pasaron, {total - exitosas} fallaron \u2716")
    print("=" * 40)

    sys.exit(0 if exitosas == total else 1)


if __name__ == "__main__":
    main()
