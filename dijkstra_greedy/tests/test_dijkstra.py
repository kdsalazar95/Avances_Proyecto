"""Pruebas del bloque Dijkstra (dijkstra_greedy/dijkstra), el sustituto determinístico
de los skills 04 a 07: adaptador del IR, mapeo, verificador y corrida en el
simulador.

No dependen de cuáles son las operaciones de operaciones.c: se recorren las
de su tabla OPERACIONES y se comparan contra la memoria final que deja el
propio C (referencia_c.py). Los detalles del adaptador y de la vectorización
se prueban con operaciones propias (test/datos/otras_operaciones.c)."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_raiz_dg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_raiz_dg), "cgra", "src"))  # simulador (cgra/src)
sys.path.insert(0, _raiz_dg)  # Dijkstra + Greedy (dijkstra_greedy/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cargador import Cargador
from cgra import CGRA
from dijkstra import (Compatibilidad, GrafoDatos, MallaMapeo, Opciones, cargar_grafo, evaluar_configuracion,
                      mapear, verificar)
from dijkstra import evaluar, grafo_vectorial
from dijkstra import evaluador as modulo_evaluador
from dijkstra.contrato import comparar
from dijkstra.evaluador import Costos, Evaluador
from dijkstra.salida import escribir_pe, simular
from formato_cgra import RUTA_POR_DEFECTO, FormatoCGRA
from malla import Malla
from programa import Programa
from referencia_c import OTRAS_OPERACIONES, Operaciones, diferencias
from run_cgra import SimuladorCGRA

FUENTE = os.path.join(os.path.dirname(_raiz_dg), "cgra", "operaciones", "operaciones.c")
CLANG = shutil.which("clang-18")
with open(RUTA_POR_DEFECTO, encoding="utf-8") as _archivo:
    CGRA_JSON = json.load(_archivo)
TIPOS = CGRA_JSON["tipos_pe"]
CGRA_3X3 = dict(CGRA_JSON, filas=3, columnas=3)  # cgra.json da rangos; esta es la malla 3x3
CGRA_3X3_C = dict(CGRA_3X3, malla=["M E M", "M V M", "M C M"])  # con un PE C (la 3x3 de cgra.json es la referencia)


def malla(*filas, tipos=TIPOS):
    letras = [fila.split() for fila in filas]
    return MallaMapeo(len(letras), len(letras[0]), letras, tipos)


def ejemplo_a_mano():
    """c[0] = (a[0] + b[0]) * 2.0"""
    return GrafoDatos.desde_dict({"regiones": {"a": [1, 1], "b": [1, 1], "c": [1, 1]}, "operaciones": [
        {"id": 0, "op": "load", "banco": "a", "indice": 0},
        {"id": 1, "op": "load", "banco": "b", "indice": 0},
        {"id": 2, "op": "fadd", "entradas": [0, 1]},
        {"id": 3, "op": "const", "valor": 2.0},
        {"id": 4, "op": "fmul", "entradas": [2, 3]},
        {"id": 5, "op": "store", "banco": "c", "indice": 0, "entradas": [4]},
    ]})


def suma_de_vectores():
    """c[0..3] = a[0..3] + b[0..3], ya vectorizado."""
    return GrafoDatos.desde_dict({"regiones": {"a": [1, 4], "b": [1, 4], "c": [1, 4]}, "operaciones": [
        {"id": 0, "op": "vload", "banco": "a", "indice": 0, "ancho": 4},
        {"id": 1, "op": "vload", "banco": "b", "indice": 0, "ancho": 4},
        {"id": 2, "op": "vadd", "entradas": [0, 1]},
        {"id": 3, "op": "vstore", "banco": "c", "indice": 0, "ancho": 4, "entradas": [2]},
    ]})


def load_add_store():
    """c[0] = a[0] + a[0]: el mismo valor entra dos veces al ADD."""
    return GrafoDatos.desde_dict({"operaciones": [
        {"id": 0, "op": "load", "banco": "a", "indice": 0},
        {"id": 1, "op": "fadd", "entradas": [0, 0]},
        {"id": 2, "op": "store", "banco": "c", "indice": 0, "entradas": [1]},
    ]})


@unittest.skipIf(CLANG is None, "no hay clang-18 para compilar operaciones.c")
class _ConOperaciones(unittest.TestCase):
    """Compila las operaciones de ``FUENTE_PRUEBA`` con el skill 02 (IR y memoria.bin
    recién generada) y guarda la memoria final que deja el C para comparar."""

    FUENTE_PRUEBA = FUENTE
    CABECERAS = None  # carpeta de memoria_cgra.h (por defecto, la de FUENTE_PRUEBA)

    @classmethod
    def setUpClass(cls):
        cls.operaciones = Operaciones(cls.FUENTE_PRUEBA, cls.CABECERAS)
        cls.nombres = tuple(cls.operaciones.nombres)
        cls.programas = str(cls.operaciones.programas)
        cls.compat = Compatibilidad.cargar()
        cls.formato = FormatoCGRA.cargar(filas=3, columnas=3)
        cls.carpetas = {op: str(cls.operaciones.carpeta(op)) for op in cls.nombres}
        cls.grafos = {op: cargar_grafo(op, cls.carpetas[op]) for op in cls.nombres}
        # La de grafo más chico: para las pruebas que solo necesitan una operación.
        cls.pequena = min(cls.nombres, key=lambda op: len(cls.grafos[op].operaciones))

    @classmethod
    def tearDownClass(cls):
        cls.operaciones.cerrar()

    def assertIgualAlC(self, operacion, obtenido):
        """La memoria ``obtenido`` debe ser la que deja el C al terminar la operación."""
        distintas = diferencias(self.operaciones.final(operacion), obtenido)
        self.assertEqual(distintas, [], f"{operacion}: {distintas[:3]}")


class _ConOtrasOperaciones(_ConOperaciones):
    """Las operaciones de prueba de test/datos/otras_operaciones.c."""

    FUENTE_PRUEBA = OTRAS_OPERACIONES
    CABECERAS = os.path.dirname(FUENTE)


def contar(grafo):
    """``{op: cantidad}`` de los nodos del grafo."""
    cuenta = {}
    for nodo in grafo.operaciones:
        cuenta[nodo.op] = cuenta.get(nodo.op, 0) + 1
    return cuenta


class TestGrafoIR(_ConOperaciones):
    """El adaptador del skill 04 desenrolla @<operacion> del IR en un grafo sin ciclos,
    para cada operación de operaciones.c."""

    def test_1_grafos_sin_ciclos_y_con_instrucciones_del_isa(self):
        """Cada grafo debe ordenarse, usar solo operaciones de la tabla y escribir al menos una región."""
        self.assertTrue(self.nombres, "operaciones.c no tiene operaciones")
        for operacion, grafo in self.grafos.items():
            with self.subTest(operacion=operacion):
                grafo.orden_topologico()
                sin_tabla = {n.op for n in grafo.operaciones} - set(self.compat.operaciones)
                self.assertEqual(sin_tabla, set())
                self.assertTrue(any(n.op == "store" for n in grafo.operaciones))

    def test_2_regiones_y_accesos_de_memoria(self):
        """Las regiones deben ser las de memoria.bin, cada acceso estar dentro de su región
        y cada load que lee algo escrito antes depender del store de esa misma posición."""
        for operacion, grafo in self.grafos.items():
            with self.subTest(operacion=operacion):
                _, formas = self.operaciones.inicial(operacion)
                self.assertEqual(grafo.regiones, formas)
                por_id = {n.id: n for n in grafo.operaciones}
                for nodo in grafo.operaciones:
                    if nodo.op in ("load", "store"):
                        filas, columnas = formas[nodo.banco]
                        self.assertLess(nodo.indice, filas * columnas)
                    for previo in (nodo.memoria if nodo.op == "load" else ()):
                        store = por_id[previo]
                        self.assertEqual((store.op, store.banco, store.indice), ("store", nodo.banco, nodo.indice))

    def test_3_el_grafo_calcula_lo_mismo_que_el_c(self):
        """Ejecutar el grafo (escalar y vectorizado) sobre memoria.bin debe dar la memoria final del C."""
        for operacion, grafo in self.grafos.items():
            inicial, _ = self.operaciones.inicial(operacion)
            self.assertIgualAlC(operacion, evaluar(grafo, inicial))
            self.assertIgualAlC(operacion, evaluar(grafo_vectorial(grafo, 4), inicial))


class TestGrafoOtrasOperaciones(_ConOtrasOperaciones):
    """El adaptador con operaciones que no son las del proyecto: reducción con producto,
    resta y división, y a*b + 1 (fmuladd) con una suma."""

    def test_1_operaciones_de_cada_grafo(self):
        """Cada grafo debe tener sus loads, operaciones y stores; fmuladd se separa en MUL + ADD."""
        cuentas = {op: contar(grafo) for op, grafo in self.grafos.items()}
        # c = a * b (12) y result *= c[i] (12): a, b, c y el result inicial; un store por c[i] y uno de result.
        self.assertEqual(cuentas["reduccion_producto"], {"load": 37, "fmul": 24, "store": 13})
        # (x - y) / 2.5 - 1: dos restas, una división y dos constantes por posición.
        self.assertEqual(cuentas["resta_division"], {"load": 32, "fsub": 32, "const": 32, "fdiv": 16, "store": 16})
        # fmuladd(a, b, 1.0) = MUL + ADD (10), y la suma de c (10).
        self.assertEqual(cuentas["mezcla"], {"load": 31, "const": 10, "fmul": 10, "fadd": 20, "store": 11})

    def test_2_cada_load_de_c_va_despues_de_su_store(self):
        """En la reducción con producto, cada c[i] que se lee depende del store de c[i]."""
        grafo = self.grafos["reduccion_producto"]
        stores_c = {n.indice: n.id for n in grafo.operaciones if n.op == "store" and n.banco == "c"}
        loads_c = [n for n in grafo.operaciones if n.op == "load" and n.banco == "c"]
        self.assertEqual(len(loads_c), 12)
        for load in loads_c:
            self.assertEqual(load.memoria, (stores_c[load.indice],))

    def test_3_el_grafo_calcula_lo_mismo_que_el_c(self):
        """El grafo escalar y el vectorizado deben dar la memoria final del C (result = producto de c)."""
        for operacion, grafo in self.grafos.items():
            inicial, _ = self.operaciones.inicial(operacion)
            self.assertIgualAlC(operacion, evaluar(grafo, inicial))
            self.assertIgualAlC(operacion, evaluar(grafo_vectorial(grafo, 4), inicial))


class TestMapeoDijkstra(unittest.TestCase):
    """Mapeo determinístico: tipos, ruteo con SEND/RECV y casos inválidos."""

    @classmethod
    def setUpClass(cls):
        cls.compat = Compatibilidad.cargar()

    def test_1_determinismo(self):
        """Mapear dos veces la misma entrada debe dar exactamente el mismo resultado."""
        malla_3x3 = malla("M E M", "M V M", "M C M")
        primero = mapear(ejemplo_a_mano(), malla_3x3, self.compat)
        segundo = mapear(ejemplo_a_mano(), malla_3x3, self.compat)
        self.assertEqual((primero.ubicacion, primero.programa, primero.metricas),
                         (segundo.ubicacion, segundo.programa, segundo.metricas))

    def test_2_ejemplo_a_mano_3x3(self):
        """(a[0] + b[0]) * 2.0 en la malla 3x3 de cgra.json: PE y ciclo de cada operación."""
        resultado = mapear(ejemplo_a_mano(), malla("M E M", "M V M", "M C M"), self.compat)
        self.assertTrue(resultado.valido, resultado.motivo)
        # Los loads salen lo más tarde posible: b de PE02 en el ciclo 0 (llega en el 1 y
        # espera), a de PE00 en el 1 (llega en el 2). El ADD va en el E (PE01) en el ciclo 3,
        # el MOV de la constante en un hueco de PE01 y el MUL justo después, sin moverse.
        self.assertEqual(resultado.ubicacion, {
            0: ("PE00", 1), 1: ("PE02", 0), 2: ("PE01", 3), 3: ("PE01", 0), 4: ("PE01", 4), 5: ("PE00", 6),
        })
        self.assertEqual(resultado.programa["PE01"], [
            "MOV rA, 2.0", "RECV east, rB", "RECV west, rC", "ADD rB, rC, rB", "MUL rA, rB, rA",
            "SEND west, rA", "NOP",
        ])
        self.assertEqual(resultado.programa["PE00"][6], "ST c[0], rA")
        self.assertEqual(resultado.metricas["ciclos"], 7)
        self.assertEqual(resultado.metricas["nops"], 63 - 12)

    def test_3_ruteo_con_relevos_1x4(self):
        """En M V V E el dato cruza los V con RECV/SEND y el ADD (con el mismo valor dos veces) queda en el ciclo 4."""
        resultado = mapear(load_add_store(), malla("M V V E"), self.compat)
        self.assertTrue(resultado.valido, resultado.motivo)
        self.assertEqual(resultado.ubicacion, {0: ("PE00", 0), 1: ("PE03", 4), 2: ("PE00", 8)})
        self.assertEqual(resultado.programa["PE01"][1:3], ["RECV west, rA", "SEND east, rA"])
        self.assertEqual(resultado.programa["PE02"][2:4], ["RECV west, rA", "SEND east, rA"])
        self.assertEqual(resultado.programa["PE03"][3:5], ["RECV west, rA", "ADD rA, rA, rA"])
        self.assertEqual(resultado.metricas["transferencias"], 6)

    def test_4_pe_de_memoria_sin_reenvio(self):
        """Si los PEs de memoria no reenvían, un dato que solo puede cruzar uno de ellos hace inválido el mapeo."""
        tipos = dict(TIPOS, S={"nombre": "solo_store", "instrucciones": ["NOP", "MOV", "ST", "SEND", "RECV"]})
        malla_1x3 = malla("M S E", tipos=tipos)
        self.assertTrue(mapear(load_add_store(), malla_1x3, self.compat).valido)
        resultado = mapear(load_add_store(), malla_1x3, self.compat, Opciones(reenvio_memoria=False))
        self.assertFalse(resultado.valido)
        self.assertIn("no se pudo rutear", resultado.motivo)

    def test_5_casos_invalidos(self):
        """Sin PE de memoria, operación sin tabla, grafo con ciclos o malla 1x1 de solo M: inválido con motivo."""
        sin_memoria = mapear(ejemplo_a_mano(), malla("E C E", "C E C"), self.compat)
        self.assertFalse(sin_memoria.valido)
        self.assertIn("ejecuta LD", sin_memoria.motivo)

        sin_tabla = GrafoDatos.desde_dict({"operaciones": [{"id": 0, "op": "load", "banco": "a", "indice": 0},
                                                           {"id": 1, "op": "frem", "entradas": [0, 0]}]})
        resultado = mapear(sin_tabla, malla("M C M"), self.compat)
        self.assertFalse(resultado.valido)
        self.assertIn("'frem' (#1) no tiene entrada en la tabla", resultado.motivo)

        con_ciclo = GrafoDatos.desde_dict({"operaciones": [{"id": 0, "op": "fadd", "entradas": [1, 1]},
                                                           {"id": 1, "op": "fadd", "entradas": [0, 0]}]})
        resultado = mapear(con_ciclo, malla("M C M"), self.compat)
        self.assertFalse(resultado.valido)
        self.assertIn("ciclos", resultado.motivo)

        datos = dict(CGRA_JSON)
        # Malla explícita: el cgra.json del simulador adapta la 1x1 a G (adaptar_mallas_pequenas).
        datos.update(filas=1, columnas=1, malla=["M"])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as archivo:
            json.dump(datos, archivo)
        try:
            resultado = evaluar_configuracion(ejemplo_a_mano(), archivo.name)
        finally:
            os.unlink(archivo.name)
        self.assertFalse(resultado.valido)
        self.assertIn("ningún PE de la malla (M) ejecuta ADD", resultado.motivo)
        self.assertIsNone(resultado.metricas["ciclos"])

    def test_6_formato_invalido_no_se_cae(self):
        """Un .json de CGRA que no existe debe dar un resultado inválido, no una excepción."""
        resultado = evaluar_configuracion(ejemplo_a_mano(), "/no/existe/cgra.json")
        self.assertFalse(resultado.valido)
        self.assertIn("formato de la CGRA inválido", resultado.motivo)


class TestVerificador(unittest.TestCase):
    """El verificador independiente acepta los mapeos válidos y detecta los alterados."""

    @classmethod
    def setUpClass(cls):
        cls.compat = Compatibilidad.cargar()
        cls.malla = malla("M E M", "M V M", "M C M")
        cls.grafo = ejemplo_a_mano()

    def mapeo(self):
        return mapear(self.grafo, self.malla, self.compat)

    def test_1_acepta_mapeos_validos(self):
        """El ejemplo a mano y el de relevos deben pasar el verificador sin errores."""
        self.assertEqual(verificar(self.grafo, self.malla, self.compat, self.mapeo()), [])
        relevos = malla("M V V E")
        self.assertEqual(verificar(load_add_store(), relevos, self.compat, mapear(load_add_store(), relevos, self.compat)), [])

    def test_2_detecta_dos_operaciones_en_el_mismo_pe_y_ciclo(self):
        """Mover el MUL al PE y ciclo del ADD debe dar un error de (PE, ciclo) repetido."""
        resultado = self.mapeo()
        resultado.ubicacion[4] = resultado.ubicacion[2]
        errores = verificar(self.grafo, self.malla, self.compat, resultado)
        self.assertTrue(any("están las dos en PE01, ciclo 3" in e for e in errores), errores)

    def test_3_detecta_un_tipo_incompatible(self):
        """Poner el ADD en un PE de memoria debe dar un error de tipo."""
        resultado = self.mapeo()
        resultado.ubicacion[2] = ("PE10", 3)
        errores = verificar(self.grafo, self.malla, self.compat, resultado)
        self.assertTrue(any("tipo M, que no tiene ADD" in e for e in errores), errores)

    def test_4_detecta_un_registro_equivocado(self):
        """Si el ADD lee un registro que no tiene su operando, la ejecución simbólica lo detecta."""
        resultado = self.mapeo()
        resultado.programa["PE01"][3] = "ADD rB, rA, rB"
        errores = verificar(self.grafo, self.malla, self.compat, resultado)
        self.assertTrue(any("PE01, ciclo 3" in e and "#2" in e for e in errores), errores)

    def test_5_detecta_un_send_sin_recv(self):
        """Un SEND cuyo vecino no hace RECV en el mismo ciclo debe ser un error."""
        resultado = self.mapeo()
        resultado.programa["PE02"][1] = "NOP"
        errores = verificar(self.grafo, self.malla, self.compat, resultado)
        self.assertTrue(any("RECV east sin SEND" in e for e in errores), errores)


class TestVectorizacion(_ConOtrasOperaciones):
    """La vectorización agrupa de a ancho_vector y el mapeo usa VLD, VADD, VSEND... en los PEs V/C."""

    def test_1_operaciones_vectorizadas(self):
        """Los grafos de 4 carriles deben tener los vload/vmul/vstore, vbcast y vredsum esperados."""
        cuentas = {}
        for operacion, grafo in self.grafos.items():
            vectorial = grafo_vectorial(grafo, 4)
            vectorial.orden_topologico()
            cuentas[operacion] = contar(vectorial)
        # c = a * b en 3 bloques; el producto de c queda escalar (VREDSUM solo suma).
        self.assertEqual(cuentas["reduccion_producto"], {"vload": 6, "vmul": 3, "vstore": 3, "load": 13,
                                                         "fmul": 12, "store": 1})
        # 4 bloques; 2.5 y 1.0 llegan a los carriles con VBCAST.
        self.assertEqual(cuentas["resta_division"], {"const": 8, "vload": 8, "vsub": 8, "vbcast": 8, "vdiv": 4,
                                                     "vstore": 4})
        # 2 bloques de c = a * b + 1 y la suma de a 4 con VREDSUM; c[8] y c[9], escalares.
        self.assertEqual(cuentas["mezcla"], {"const": 4, "vload": 6, "vmul": 2, "vbcast": 2, "vadd": 2,
                                             "vstore": 2, "load": 7, "fmul": 2, "fadd": 6, "store": 3,
                                             "vredsum": 2})

    def test_2_todos_los_vectores_tienen_el_ancho(self):
        """Ningún vload ni vstore debe pasar de ancho_vector."""
        for grafo in self.grafos.values():
            for nodo in grafo_vectorial(grafo, 4).operaciones:
                if nodo.op in ("vload", "vstore"):
                    self.assertLessEqual(nodo.ancho, 4)

    def test_3_suma_de_vectores_en_m_v_m(self):
        """En M V M los vectores van de M al V con VSEND/VRECV, el VADD en el V y el VST de vuelta en una M."""
        resultado = mapear(suma_de_vectores(), malla("M V M"), self.compat)
        self.assertTrue(resultado.valido, resultado.motivo)
        self.assertEqual(resultado.ubicacion[2], ("PE01", 3))
        programa = resultado.programa
        self.assertIn("VADD v0, v1, v0", programa["PE01"])
        self.assertTrue(any(i.startswith("VLD") for i in programa["PE00"] + programa["PE02"]))
        self.assertTrue(any(i.startswith("VST c[0]") for i in programa["PE00"] + programa["PE02"]))
        self.assertEqual(sum(i.startswith("VRECV") for i in programa["PE01"]), 2)
        self.assertEqual(verificar(suma_de_vectores(), malla("M V M"), self.compat, resultado), [])

    def test_4_un_e_no_reenvia_vectores(self):
        """En M E V el vector tendría que cruzar el E, que no tiene VSEND/VRECV: inválido en modo vectorial."""
        resultado = mapear(suma_de_vectores(), malla("M E V"), self.compat)
        self.assertFalse(resultado.valido)
        self.assertIn("no se pudo rutear", resultado.motivo)

    def test_5_auto_vuelve_a_escalar(self):
        """Con modo auto, si los vectores no tienen ruta, el mapeo debe ser el escalar (y válido)."""
        formato = FormatoCGRA(1, 3, 4096, TIPOS, [["M", "E", "V"]], 4)
        vectorial = evaluar_configuracion(self.grafos["resta_division"], formato, self.compat, modo="vectorial")
        self.assertFalse(vectorial.valido)
        auto = evaluar_configuracion(self.grafos["resta_division"], formato, self.compat)
        self.assertTrue(auto.valido, auto.motivo)
        self.assertEqual(auto.metricas["modo"], "escalar")

    def test_6_verificador_detecta_vsend_con_recv(self):
        """Un VRECV cambiado por RECV debe ser un error (un vector no viaja por SEND/RECV)."""
        resultado = mapear(suma_de_vectores(), malla("M V M"), self.compat)
        ciclo = next(c for c, i in enumerate(resultado.programa["PE01"]) if i.startswith("VRECV"))
        resultado.programa["PE01"][ciclo] = resultado.programa["PE01"][ciclo].replace("VRECV", "RECV")
        errores = verificar(suma_de_vectores(), malla("M V M"), self.compat, resultado)
        self.assertTrue(any("sin VRECV" in e or "no es un registro escalar" in e for e in errores), errores)


class _Simular:
    """Corre un mapeo en el simulador (para las clases con operaciones compiladas)."""

    def correr(self, operacion, formato, modo="auto"):
        resultado = evaluar_configuracion(self.grafos[operacion], formato, self.compat, modo=modo)
        self.assertTrue(resultado.valido, resultado.motivo)
        malla_mapeo = MallaMapeo.desde_formato(formato)
        with tempfile.TemporaryDirectory() as directorio:
            escribir_pe(resultado, malla_mapeo, directorio)
            correcto, distintas, ciclos = simular(
                formato, os.path.join(self.carpetas[operacion], "memoria.bin"), directorio, self.grafos[operacion])
        self.assertTrue(correcto, distintas[:3])
        self.assertEqual(ciclos, resultado.metricas["ciclos"])
        return resultado

    def memoria_simulada(self, operacion, formato, modo="auto"):
        """``{región: valores}`` que deja el simulador con el mapeo de la operación."""
        resultado = evaluar_configuracion(self.grafos[operacion], formato, self.compat, modo=modo)
        self.assertTrue(resultado.valido, resultado.motivo)
        with tempfile.TemporaryDirectory() as directorio:
            escribir_pe(resultado, MallaMapeo.desde_formato(formato), directorio)
            cgra = CGRA(formato)
            programa = Cargador(cgra).cargar(os.path.join(self.carpetas[operacion], "memoria.bin"), directorio)
            cgra.ejecutar(programa)
        return {region: cgra.memoria.region(region) for region in self.grafos[operacion].regiones}


class TestSimuladorDijkstra(_Simular, _ConOperaciones):
    """Los PE*.txt del bloque corren en el simulador y dejan en memoria lo que calcula el C."""

    def test_1_todas_las_operaciones_en_cgra_json(self):
        """Cada operación de operaciones.c en la malla 3x3 de cgra.json, escalar y vectorial: memoria correcta."""
        for operacion in self.nombres:
            for modo in ("escalar", "vectorial"):
                with self.subTest(operacion=operacion, modo=modo):
                    resultado = self.correr(operacion, self.formato, modo)
                    self.assertEqual(resultado.metricas["modo"], modo)
                    self.assertEqual(resultado.metricas["slots"], resultado.metricas["ciclos"] * 9)

    def test_2_resultado_igual_al_del_c(self):
        """La memoria que deja el simulador debe ser la que deja el C, en cada operación."""
        for operacion in self.nombres:
            with self.subTest(operacion=operacion):
                self.assertIgualAlC(operacion, self.memoria_simulada(operacion, self.formato))

    def test_3_otras_mallas(self):
        """Mallas de otro tamaño y mapas explícitos con G, M, E, V y C también deben dar la memoria correcta."""
        mallas = [
            (1, 3, [["M", "E", "M"]]),
            (2, 2, [["G", "V"], ["E", "M"]]),
            (4, 4, [["M", "E", "C", "M"], ["G", "V", "V", "M"], ["M", "C", "E", "G"], ["V", "M", "C", "E"]]),
        ]
        for filas, columnas, mapa in mallas:
            formato = FormatoCGRA(filas, columnas, 4096, TIPOS, mapa, 4)
            with self.subTest(malla=mapa):
                self.correr(self.pequena, formato)

    def test_4_pe_txt_en_formato_del_skill_08(self):
        """Debe escribir un PE*.txt por PE (borrando los viejos) que Programa.leer() acepta tal cual."""
        resultado = evaluar_configuracion(self.grafos[self.pequena], self.formato, self.compat)
        with tempfile.TemporaryDirectory() as directorio:
            open(os.path.join(directorio, "PE99.txt"), "w").close()
            open(os.path.join(directorio, "notas.txt"), "w").close()
            escribir_pe(resultado, MallaMapeo.desde_formato(self.formato), directorio)
            nombres = sorted(os.listdir(directorio))
            programa = Programa.leer(directorio, Malla(3, 3))
            with open(os.path.join(directorio, "PE00.txt"), encoding="utf-8") as archivo:
                primera = archivo.readline().rstrip("\n")
        self.assertEqual(nombres, [f"PE{f}{c}.txt" for f in range(3) for c in range(3)] + ["notas.txt"])
        self.assertEqual(len(programa.ciclos()), resultado.metricas["ciclos"])
        self.assertEqual(primera, "// PE00 instruction list")


class TestSimuladorOtrasOperaciones(_Simular, _ConOtrasOperaciones):
    """Operaciones que no son las del proyecto también corren bien en el simulador."""

    def test_1_memoria_igual_a_la_del_c(self):
        """Reducción con producto, resta y división, y fmuladd con suma: escalar y vectorial, como el C."""
        for operacion in self.nombres:
            for modo in ("escalar", "vectorial"):
                with self.subTest(operacion=operacion, modo=modo):
                    self.correr(operacion, self.formato, modo)
                    self.assertIgualAlC(operacion, self.memoria_simulada(operacion, self.formato, modo))

    def test_2_vectorial_mas_rapido(self):
        """En resta_division (16 posiciones independientes) el mapeo vectorial debe tardar menos ciclos."""
        escalar = self.correr("resta_division", self.formato, "escalar")
        vectorial = self.correr("resta_division", self.formato, "vectorial")
        self.assertLess(vectorial.metricas["ciclos"], escalar.metricas["ciclos"])


class TestEvaluador(_ConOperaciones):
    """El evaluador sigue el contrato con la solución RL: mismos estados, validaciones y costo normalizado."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.evaluador = Evaluador(cls.nombres, carpeta_programas=cls.programas)
        cls.costos = cls.evaluador.costos
        cls.norm = cls.evaluador.normalizadores()

    def costo_esperado(self, evaluacion, k_a=1, k_l=1, k_r=1):
        por_op = [k_a * evaluacion.area / self.norm["operaciones"][op]["C_A_ref"]
                  + k_l * e.C_L / self.norm["operaciones"][op]["C_L_ref"] + k_r * e.C_R
                  for op, e in evaluacion.operaciones.items()]
        return sum(por_op) / len(por_op)

    def test_1_costos_de_cgra_json(self):
        """En la 3x3 M E M / M V M / M C M: C_A = 6 M x 12 + E 1 + V 2 + C 3 = 78, C_L = ciclos·(1 + NOPs/(ciclos·PEs))
        por operación, C_R = 0 y C = promedio de C_A/C_A_ref + C_L/C_L_ref + C_R con los normalizadores del RL."""
        evaluacion = self.evaluador.evaluar(CGRA_3X3_C)
        self.assertEqual(evaluacion.estado, "correcta", evaluacion.motivo)
        self.assertEqual(evaluacion.area, 78)
        self.assertEqual(evaluacion.metricas["capacidades"], {"MS": 6, "MV": 6, "CS": 2, "CV": 2})
        for e in evaluacion.operaciones.values():
            self.assertAlmostEqual(e.C_L, e.ciclos * (1 + e.nops / (e.ciclos * 9)))
            self.assertEqual(e.C_R, 0)
            self.assertEqual(e.validacion["elementos_fallidos"], 0)
        self.assertEqual(evaluacion.metricas["ciclos"], sum(e.ciclos for e in evaluacion.operaciones.values()))
        self.assertAlmostEqual(evaluacion.costo, self.costo_esperado(evaluacion))
        self.assertEqual({e.modo for e in evaluacion.operaciones.values()}, {"vectorial"})

    def test_2_normalizadores_congelados(self):
        """Las operaciones de operaciones.c usan los C_L_ref congelados de la corrida RL, no los de Dijkstra."""
        congelados = json.loads(self.costos.ruta_normalizadores.read_text(encoding="utf-8"))
        for op in self.nombres:
            self.assertEqual(self.norm["operaciones"][op]["C_L_ref"], congelados["operaciones"][op]["C_L_ref"])
        self.assertEqual(self.norm["C_A_ref"], 76)

    def test_3_normalizadores_medidos_dan_costo_2(self):
        """Sin normalizadores fijos se miden con la referencia: la referencia cuesta K_A + K_L = 2."""
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas, normalizadores=None)
        evaluacion = evaluador.evaluar(evaluador.referencia())
        self.assertEqual(evaluacion.estado, "correcta", evaluacion.motivo)
        self.assertAlmostEqual(evaluacion.costo, 2.0)

    def test_4_referencia_que_no_sirve(self):
        """Si la referencia no es correcta no hay normalización: ErrorReferencia, como en el RL."""
        costos = Costos.cargar()
        costos.referencia = ["M V M"]  # V no hace ADD escalar: Dijkstra no la puede mapear
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas, costos=costos, normalizadores=None)
        with self.assertRaises(modulo_evaluador.ErrorReferencia):
            evaluador.evaluar(CGRA_3X3)

    def test_5_accion_invalida(self):
        """Fuera de rango, menos de 3 columnas, M en el interior, G o un tipo sin definir: accion_invalida,
        costo P_fallo, sin mapear ni simular."""
        casos = {
            "filas = 40 fuera del rango permitido [1, 32]": dict(CGRA_JSON, filas=40, columnas=3),
            "se necesitan al menos 3 columnas": dict(CGRA_JSON, filas=1, columnas=1),
            "PE(1,1) es M y accede a memoria": dict(CGRA_3X3, malla=["M E M", "M M M", "M C M"]),
            "PE(0,1) es G y accede a memoria": dict(CGRA_3X3, malla=["M G M", "M V M", "M C M"]),
            "PE(0,0) es E: los bordes": dict(CGRA_3X3, malla=["E E M", "M V M", "M C M"]),
            "'X' no está definido": dict(CGRA_3X3, malla=["M X M", "M V M", "M C M"]),
        }
        with mock.patch.object(modulo_evaluador, "evaluar_configuracion", side_effect=AssertionError):
            for texto, configuracion in casos.items():
                evaluacion = self.evaluador.evaluar(configuracion)
                self.assertEqual(evaluacion.estado, "accion_invalida", texto)
                self.assertEqual(evaluacion.costo, self.costos.p_fallo)
                self.assertIn(texto, evaluacion.motivo)
                self.assertNotIn("/tmp", evaluacion.motivo)

    def test_6_fallo_evaluador(self):
        """Si Dijkstra no puede mapear, o sus PE*.txt no pasan las reglas de E03, es fallo_evaluador."""
        sin_add = dict(CGRA_3X3, filas=1, malla=["M V M"])
        evaluacion = self.evaluador.evaluar(sin_add)
        self.assertEqual(evaluacion.estado, "fallo_evaluador")
        self.assertEqual(evaluacion.costo, self.costos.p_fallo)
        error = {"skill": "E03", "regla": "salida_sin_cubrir", "detalle": "1 de 1 posiciones de 'x'"}
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas)
        with mock.patch.object(modulo_evaluador, "validar_programas",
                               return_value={"valida": False, "errores": [error]}):
            evaluacion = evaluador.evaluar(CGRA_3X3)
        self.assertEqual(evaluacion.estado, "fallo_evaluador")
        self.assertEqual(evaluacion.errores[0]["regla"], "salida_sin_cubrir")

    def test_7_fallo_simulador_por_tiempo(self):
        """El simulador corre en otro proceso con t_max_simulacion_s: si lo supera, fallo_simulador."""
        costos = Costos.cargar()
        costos.t_max_simulacion_s = 0.001
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas, costos=costos)
        evaluacion = evaluador.evaluar(CGRA_3X3)
        self.assertEqual(evaluacion.estado, "fallo_simulador")
        self.assertEqual(evaluacion.errores[0]["regla"], "tiempo_excedido")
        self.assertEqual(evaluacion.costo, costos.p_fallo)

    def test_8_resultado_equivocado(self):
        """Un elemento fuera de tolerancia hace la configuración incorrecta (costo P_fallo) y C_R es el
        promedio de |y − ŷ|/(|y| + ε): sobrestimar no premia."""
        simular = modulo_evaluador.simular
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas)
        region, posicion = next((r, i) for r in evaluador.salidas[self.pequena]
                                for i, v in enumerate(self.operaciones.final(self.pequena)[r]) if v)
        total = sum(len(self.operaciones.final(self.pequena)[r]) for r in evaluador.salidas[self.pequena])

        def alterado(factor):
            def simular_alterado(*argumentos):
                datos, error = simular(*argumentos)
                datos["final"][region][posicion] *= factor
                return datos, error
            return simular_alterado

        for factor in (1.5, 0.5):
            evaluador.guardadas.clear()
            evaluador.mediciones.clear()
            with mock.patch.object(modulo_evaluador, "simular", alterado(factor)):
                evaluacion = evaluador.evaluar(CGRA_3X3)
            self.assertEqual(evaluacion.estado, "incorrecta")
            self.assertEqual(evaluacion.costo, self.costos.p_fallo)
            self.assertAlmostEqual(evaluacion.correctitud, 0.5 / total, places=6)
            self.assertEqual(evaluacion.operaciones[self.pequena].validacion["elementos_fallidos"], 1)
        # Dentro de la tolerancia sigue siendo correcta, con un C_R pequeño que sí cuenta en el costo.
        evaluador.guardadas.clear()
        evaluador.mediciones.clear()
        with mock.patch.object(modulo_evaluador, "simular", alterado(1 + 5e-7)):
            evaluacion = evaluador.evaluar(CGRA_3X3)
        self.assertEqual(evaluacion.estado, "correcta")
        self.assertGreater(evaluacion.correctitud, 0)

    def test_9_misma_configuracion_no_se_repite(self):
        """Evaluar dos veces la misma configuración debe devolver lo guardado, sin volver a mapear."""
        primera = self.evaluador.evaluar(CGRA_3X3)
        with mock.patch.object(modulo_evaluador, "evaluar_configuracion", side_effect=AssertionError):
            segunda = self.evaluador.evaluar(dict(CGRA_3X3))
        self.assertIs(primera, segunda)

    def test_10_latencia_en_ciclos_y_pesos(self):
        """Con formula_latencia = 'ciclos', C_L son los ciclos; los K multiplican los términos normalizados."""
        costos = Costos.cargar()
        costos.formula_latencia = "ciclos"
        costos.ruta_normalizadores = None
        costos.K = {"K_A": 3, "K_L": 2, "K_R": 1}
        evaluador = Evaluador(self.nombres, carpeta_programas=self.programas, costos=costos, normalizadores=None)
        evaluacion = evaluador.evaluar(CGRA_3X3_C)
        norm = evaluador.normalizadores()
        for op, e in evaluacion.operaciones.items():
            self.assertEqual(e.C_L, e.ciclos)
        por_op = [3 * 78 / norm["C_A_ref"] + 2 * e.C_L / norm["operaciones"][op]["C_L_ref"]
                  for op, e in evaluacion.operaciones.items()]
        self.assertAlmostEqual(evaluacion.costo, sum(por_op) / len(por_op))

    def test_11_normalizadores_de_otra_formula(self):
        """Los normalizadores congelados se midieron con una fórmula: no se usan con otra."""
        costos = Costos.cargar()
        costos.formula_latencia = "ciclos"
        with self.assertRaises(ValueError):
            Evaluador([self.pequena], carpeta_programas=self.programas, costos=costos)

    def test_12_cotas(self):
        """La cota por área (K_A·C_A/C_A_ref) no supera el costo de una correcta."""
        tipos = CGRA_JSON["tipos_pe"]
        for configuracion in (CGRA_3X3, dict(CGRA_JSON, filas=2, columnas=4)):
            evaluacion = self.evaluador.evaluar(configuracion)
            self.assertEqual(evaluacion.estado, "correcta", evaluacion.motivo)
            cota = self.evaluador.cota(configuracion)
            self.assertLessEqual(cota, evaluacion.costo)
            self.assertAlmostEqual(cota, evaluacion.area / 76)
        self.assertAlmostEqual(self.evaluador.cota_tamano(tipos, 9), (12 + 8 * 1) / 76)

    def test_13_comparacion_funcional(self):
        """Validación funcional del RL: todos los elementos dentro de eps_abs + eps_rel·|y|, C_R con valor absoluto."""
        tolerancias = {"eps_abs": 1e-6, "eps_rel": 1e-6, "epsilon_cr": 1e-9}
        iguales = comparar({"r": [10.0, -4.0]}, {"r": [10.0, -4.0]}, tolerancias)
        self.assertTrue(iguales["aprobada"])
        self.assertEqual(iguales["C_R"], 0)
        arriba = comparar({"r": [10.0, 4.0]}, {"r": [15.0, 4.0]}, tolerancias)
        abajo = comparar({"r": [10.0, 4.0]}, {"r": [5.0, 4.0]}, tolerancias)
        self.assertFalse(arriba["aprobada"])
        self.assertAlmostEqual(arriba["C_R"], abajo["C_R"])
        self.assertAlmostEqual(arriba["C_R"], 0.5 / 2, places=6)
        no_finito = comparar({"r": [1.0]}, {"r": [float("inf")]}, tolerancias)
        self.assertFalse(no_finito["aprobada"])
        self.assertIsNone(no_finito["C_R"])
        faltante = comparar({"r": [1.0]}, {}, tolerancias)
        self.assertEqual(faltante["errores_estructurales"], ["falta la región 'r' en la memoria final"])


if __name__ == "__main__":
    unittest.main()
