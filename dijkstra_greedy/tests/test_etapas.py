"""Contratos entre las etapas independientes, mediante carpetas de evaluación y JSON.

La simulación es la del simulador de cgra/src/ (el del RL): reporte_ciclos.txt
y la memoria final con --write-back."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from test_dijkstra import _ConOperaciones, CGRA_3X3, CGRA_JSON
from dijkstra import SRC_SIMULADOR, cargar_grafo, evaluar_configuracion
from dijkstra.contrato import Costos
from dijkstra.hardware import MallaMapeo
from dijkstra.salida import escribir_pe
from dijkstra.simulacion import ARCHIVO_MEMORIA, ARCHIVO_REPORTE, CARPETA_PE, comando, preparar
from dijkstra.evaluador import Evaluador, configuracion_de
from memoria import ImagenMemoria
from evaluar_cgra import evaluar_reportes
from agente_greedy import avanzar
from dijkstra.agente import Parametros

SRC = Path(__file__).resolve().parent.parent  # dijkstra_greedy/
OTRA = dict(CGRA_3X3, malla=["M E M", "M V M", "M C M"])
REFERENCIA = dict(CGRA_3X3, malla=["M E M", "M V M", "M E M"])


class TestEtapasSeparadas(_ConOperaciones):
    def carpeta(self, directorio, configuracion=OTRA, nombre="otra"):
        """Mapea con Dijkstra y corre cgra/src/run_cgra.py con --write-back, como en README.md."""
        formato = Evaluador.formato(configuracion)
        grafo = cargar_grafo(self.pequena, Path(self.programas) / self.pequena)
        resultado = evaluar_configuracion(grafo, formato)
        carpeta = Path(directorio) / nombre
        preparar(carpeta, configuracion_de(formato), Path(self.programas) / self.pequena / "memoria.bin")
        escribir_pe(resultado, MallaMapeo.desde_formato(formato), carpeta / CARPETA_PE)
        subprocess.run(comando(carpeta), cwd=SRC_SIMULADOR, check=True, capture_output=True, timeout=60)
        return grafo, carpeta

    def evaluar(self, reportes, **opciones):
        return evaluar_reportes(reportes, Costos.cargar(), carpeta_programas=self.programas, **opciones)

    def test_costos_desde_reporte_sin_mapear_ni_simular(self):
        """El evaluador toma los ciclos del reporte del simulador sin invocar mapeo o ejecución."""
        with tempfile.TemporaryDirectory() as tmp:
            _, carpeta = self.carpeta(tmp)
            reporte = carpeta / ARCHIVO_REPORTE
            texto = reporte.read_text(encoding="utf-8")
            ciclos = int(re.search(r"^Total de ciclos:\s+(\d+)", texto, re.M).group(1)) + 1
            # Una medición externa distinta debe respetarse: no recalcular desde el mapeador.
            reporte.write_text(re.sub(r"^(Total de ciclos:\s+)\d+", rf"\g<1>{ciclos}", texto, flags=re.M),
                               encoding="utf-8")
            with mock.patch('dijkstra.evaluador.evaluar_configuracion', side_effect=AssertionError), \
                 mock.patch('dijkstra.evaluador.simular', side_effect=AssertionError):
                salida = self.evaluar({self.pequena: carpeta})
        e = salida['evaluacion']
        op = e['operaciones'][self.pequena]
        self.assertEqual(e['estado'], 'correcta')
        self.assertEqual(op['ciclos'], ciclos)
        self.assertAlmostEqual(op['C_L'], ciclos * (1 + op['nops'] / (ciclos * 9)))
        self.assertEqual(e['metricas']['C_R'], 0)
        c_l_ref = e['normalizadores']['operaciones'][self.pequena]['C_L_ref']
        self.assertAlmostEqual(e['costo'], 78 / 76 + op['C_L'] / c_l_ref)

    def test_salida_equivocada_se_penaliza_en_greedy(self):
        """Una salida fuera de tolerancia es incorrecta: cuesta P_fallo y Greedy la registra con r = −P_fallo."""
        with tempfile.TemporaryDirectory() as tmp:
            grafo, ref = self.carpeta(tmp, REFERENCIA, "referencia")
            _, carpeta = self.carpeta(tmp)
            regiones, formas = ImagenMemoria.leer(carpeta / ARCHIVO_MEMORIA).regiones()
            region, indice = next((n.banco, n.indice) for n in grafo.operaciones
                                  if n.op == 'store' and regiones[n.banco][n.indice] != 0)
            regiones[region][indice] *= 1.5
            ImagenMemoria.desde_regiones(regiones, formas).guardar(carpeta / ARCHIVO_MEMORIA)
            referencia = self.evaluar({self.pequena: ref})
            salida = self.evaluar({self.pequena: carpeta})
        self.assertEqual(salida['evaluacion']['estado'], 'incorrecta')
        self.assertEqual(salida['evaluacion']['costo'], 10.0)
        self.assertGreater(salida['evaluacion']['metricas']['C_R'], 0)
        with mock.patch.object(Evaluador, 'evaluar', side_effect=AssertionError):
            estado, propuestas, entradas = avanzar(CGRA_JSON, Parametros(lado_maximo=3), {}, [referencia, salida])
        self.assertEqual([e['id_evaluacion'] for e in entradas], ['referencia', 'it-0001'])
        self.assertEqual(entradas[0]['recompensa'], 0.0)
        self.assertEqual(entradas[1]['recompensa'], -10.0)
        self.assertEqual(entradas[1]['detalle_recompensa']['C_best_prev'], referencia['evaluacion']['costo'])
        self.assertEqual(len(estado['cadena']), 1)
        self.assertEqual(estado['iteraciones_sin_mejora'], 1)
        self.assertTrue(propuestas)

    def test_greedy_empieza_por_la_referencia_y_respeta_el_presupuesto(self):
        """Sin evaluaciones propone solo la referencia; no acepta otra primera; con max_evaluaciones
        agotado termina sin propuestas."""
        estado, propuestas, _ = avanzar(CGRA_JSON, Parametros(), {}, [])
        self.assertEqual([p['malla'] for p in propuestas], [REFERENCIA['malla']])
        with tempfile.TemporaryDirectory() as tmp:
            _, ref = self.carpeta(tmp, REFERENCIA, "referencia")
            _, carpeta = self.carpeta(tmp)
            referencia = self.evaluar({self.pequena: ref})
            otra = self.evaluar({self.pequena: carpeta})
        with self.assertRaises(ValueError):
            avanzar(CGRA_JSON, Parametros(), {}, [otra])
        estado, propuestas, _ = avanzar(CGRA_JSON, Parametros(max_evaluaciones=3), {}, [referencia])
        self.assertEqual(len(propuestas), 3)
        estado, propuestas, entradas = avanzar(CGRA_JSON, Parametros(max_evaluaciones=1), {}, [referencia, otra])
        self.assertEqual(propuestas, [])
        self.assertEqual(estado['estado_corrida'], 'terminada: max_evaluaciones')
        self.assertAlmostEqual(entradas[1]['recompensa'], referencia['evaluacion']['costo'] - otra['evaluacion']['costo'])
        self.assertTrue(all(o['estado_evaluacion'] == 'correcta' for o in entradas))

    def test_fallos_de_etapa(self):
        """fallo.json de mapear_dijkstra.py y --fallo del simulador dan fallo_evaluador y fallo_simulador;
        una configuración fuera del espacio de acciones es accion_invalida."""
        with tempfile.TemporaryDirectory() as tmp:
            cgra = Path(tmp) / "sin_add.json"
            cgra.write_text(json.dumps(dict(CGRA_3X3, filas=1, malla=["M V M"])))
            proceso = subprocess.run([sys.executable, str(SRC / "mapear_dijkstra.py"), self.pequena, "--cgra",
                                      str(cgra), "--salida", str(Path(tmp) / "eval")], capture_output=True,
                                     text=True, cwd=SRC)
            self.assertEqual(proceso.returncode, 1)
            self.assertTrue((Path(tmp) / "eval" / "fallo.json").is_file(), proceso.stdout + proceso.stderr)
            salida = evaluar_reportes({self.pequena: Path(tmp) / "eval"}, Costos.cargar())
        self.assertEqual(salida['evaluacion']['estado'], 'fallo_evaluador')
        self.assertEqual(salida['evaluacion']['costo'], 10.0)
        simulador = evaluar_reportes({}, Costos.cargar(), {self.pequena: ("fallo_simulador", "tiempo")},
                                     configuracion=OTRA)
        self.assertEqual(simulador['evaluacion']['estado'], 'fallo_simulador')
        invalida = evaluar_reportes({}, Costos.cargar(), {self.pequena: ("fallo_simulador", "x")},
                                    configuracion=dict(CGRA_3X3, malla=["M E M", "M M M", "M C M"]))
        self.assertEqual(invalida['evaluacion']['estado'], 'accion_invalida')

    def test_mapear_arma_la_carpeta_de_evaluacion(self):
        """mapear_dijkstra.py deja cgra.json, memoria.bin y pe_instructions/ listos para cgra/src/run_cgra.py."""
        with tempfile.TemporaryDirectory() as tmp:
            cgra = Path(tmp) / "cgra.json"
            cgra.write_text(json.dumps(OTRA))
            proceso = subprocess.run([sys.executable, str(SRC / "mapear_dijkstra.py"), "todas", "--cgra", str(cgra),
                                      "--salida", str(Path(tmp) / "eval")], capture_output=True, text=True, cwd=SRC)
            self.assertEqual(proceso.returncode, 0, proceso.stdout + proceso.stderr)
            for op in ("reduccion", "matmul", "convolucion"):
                carpeta = Path(tmp) / "eval" / op
                self.assertEqual(sorted(p.name for p in carpeta.iterdir()), ["cgra.json", "memoria.bin", CARPETA_PE])
                self.assertEqual(len(list((carpeta / CARPETA_PE).glob("PE*.txt"))), 9)

    def test_normalizadores_de_la_referencia(self):
        """Sin C_L_ref de una operación, solo la referencia correcta los fija (costo 2); otra configuración no."""
        with tempfile.TemporaryDirectory() as tmp:
            _, ref = self.carpeta(tmp, REFERENCIA, "referencia")
            _, carpeta = self.carpeta(tmp)
            referencia = self.evaluar({self.pequena: ref}, normalizadores=None)
            self.assertAlmostEqual(referencia['evaluacion']['costo'], 2.0)
            with self.assertRaises(ValueError):
                self.evaluar({self.pequena: carpeta}, normalizadores=None)
            otra = self.evaluar({self.pequena: carpeta}, normalizadores=referencia['evaluacion']['normalizadores'])
        self.assertEqual(otra['evaluacion']['estado'], 'correcta')

    def test_carpeta_sin_simular(self):
        """Una carpeta que no pasó por el simulador (sin reporte_ciclos.txt) no se convierte en evaluación."""
        with tempfile.TemporaryDirectory() as tmp:
            _, carpeta = self.carpeta(tmp)
            (carpeta / ARCHIVO_REPORTE).unlink()
            with self.assertRaises(ValueError):
                self.evaluar({self.pequena: carpeta})
