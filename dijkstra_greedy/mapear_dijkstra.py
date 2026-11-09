"""Etapa Dijkstra: mapear IR existente y exportar instrucciones, sin simular.

    python3 mapear_dijkstra.py todas --cgra config.json --salida resultados/prueba_001

``--salida`` es la carpeta de evaluación de la operación (con ``todas``, una por
operación en ``<salida>/<operacion>/``), lista para el simulador de cgra/src/:
``cgra.json``, copia de ``memoria.bin`` y ``pe_instructions/PE*.txt``. Si Dijkstra
no puede mapear o los .txt no pasan E03, deja ``fallo.json`` en su lugar.
"""

import argparse
import json
from pathlib import Path

from dijkstra import (CARPETA_PROGRAMAS, MODOS, RUTA_CGRA_JSON, ErrorIR, Opciones, cargar_grafo,
                      evaluar_configuracion, operaciones as operaciones_del_c)
from dijkstra.contrato import FALLO_EVALUADOR
from dijkstra.evaluador import configuracion_de
from dijkstra.hardware import MallaMapeo
from dijkstra.salida import escribir_pe
from dijkstra.simulacion import ARCHIVO_CGRA, ARCHIVO_MEMORIA, CARPETA_PE, preparar
from dijkstra.validacion_programas import validar_programas
from formato_cgra import FormatoCGRA


def correr(operacion, formato, opciones, salida=None, simular_programa=False, modo="auto"):
    carpeta = CARPETA_PROGRAMAS / operacion
    try:
        grafo = cargar_grafo(operacion, carpeta)
    except (ErrorIR, OSError, ValueError) as error:
        print(f"{operacion:<12} error  no se pudo leer el grafo: {error}")
        return False
    resultado = evaluar_configuracion(grafo, formato, opciones=opciones, modo=modo)
    if not resultado.valido:
        print(f"{operacion:<12} inválido  {resultado.motivo}")
        _fallo(salida, formato, resultado.motivo,
               [{"skill": "Dijkstra", "regla": "sin_mapeo", "detalle": resultado.motivo}])
        return False
    m = resultado.metricas
    print(f"{operacion:<12} válido ({m['modo']})  {m['operaciones']} operaciones, {m['ciclos']} ciclos, "
          f"{m['transferencias']} transferencias, NOPs {m['nops']}/{m['slots']} ({m['ratio_nops']:.1%})")
    if simular_programa:
        raise ValueError("La simulación se ejecuta por separado con run_cgra.py")
    if salida is None:
        return True
    malla = MallaMapeo.desde_formato(formato)
    salida = Path(salida)
    (salida / ARCHIVO_FALLO).unlink(missing_ok=True)
    preparar(salida, configuracion_de(formato), carpeta / "memoria.bin")
    directorio = escribir_pe(resultado, malla, salida / CARPETA_PE)
    # E03: las mismas reglas con que el RL valida los .txt de los skills.
    regiones = sorted({n.banco for n in grafo.operaciones if n.op == "store"})
    e03 = validar_programas(formato, directorio, carpeta / "memoria.bin", regiones)
    if not e03["valida"]:
        _limpiar(salida)
        motivo = "; ".join(e["detalle"] for e in e03["errores"][:5])
        print(f"{'':<12} los PE*.txt no pasan E03: {motivo}")
        _fallo(salida, formato, motivo, e03["errores"])
        return False
    print(f"{'':<12} PE*.txt en {directorio} (E03 aprobada)")
    return True


def _limpiar(salida):
    """Una carpeta con fallo no se simula: sin cgra.json, memoria ni PE*.txt."""
    for nombre in (ARCHIVO_CGRA, ARCHIVO_MEMORIA):
        (Path(salida) / nombre).unlink(missing_ok=True)
    pe = Path(salida) / CARPETA_PE
    if pe.is_dir():
        for archivo in pe.glob("PE*.txt"):
            archivo.unlink()


ARCHIVO_FALLO = "fallo.json"


def _fallo(salida, formato, motivo, errores):
    """Deja ``<salida>/fallo.json`` para evaluar_cgra.py (estado fallo_evaluador)."""
    if salida is None:
        return
    Path(salida).mkdir(parents=True, exist_ok=True)
    _limpiar(salida)
    datos = dict(estado=FALLO_EVALUADOR, motivo=motivo, errores=errores, configuracion=configuracion_de(formato))
    ruta = Path(salida) / ARCHIVO_FALLO
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'':<12} fallo_evaluador en {ruta}: la carpeta va igual a evaluar_cgra.py con --reporte")


def main():
    parser = argparse.ArgumentParser(description="Bloque Dijkstra: mapea una operación en la CGRA (skills 04-07).")
    parser.add_argument("operacion", help="Operación de programas/ (ej. matmul) o 'todas'")
    parser.add_argument("--cgra", metavar="JSON", default=RUTA_CGRA_JSON, help="Formato de la CGRA")
    parser.add_argument("--filas", type=int, help="Filas de la malla, dentro del rango del formato")
    parser.add_argument("--columnas", type=int, help="Columnas de la malla, dentro del rango del formato")
    parser.add_argument("--salida", metavar="DIR", help="Carpeta de evaluación: cgra.json, memoria.bin y pe_instructions/ (con 'todas', DIR/<operacion>)")
    parser.add_argument("--modo", choices=MODOS, default="auto",
                        help="auto: vectorial si la CGRA puede, si no escalar; mejor: prueba los dos")
    parser.add_argument("--sin-reenvio-memoria", action="store_true",
                        help="Los PEs de memoria no reenvían datos que no cargaron ellos")
    argumentos = parser.parse_args()
    try:
        formato = FormatoCGRA.cargar(argumentos.cgra, argumentos.filas, argumentos.columnas)
    except ValueError as error:
        raise SystemExit(f"Formato de la CGRA inválido: {error}")
    opciones = Opciones(reenvio_memoria=not argumentos.sin_reenvio_memoria)
    operaciones = operaciones_del_c() if argumentos.operacion == "todas" else [argumentos.operacion]
    print(f"CGRA {formato.filas}x{formato.columnas}: {' | '.join(formato.mapa())}")
    correctas = True
    for operacion in operaciones:
        salida = argumentos.salida
        if salida and argumentos.operacion == "todas":
            salida = Path(salida) / operacion
        correctas &= correr(operacion, formato, opciones, salida, False, argumentos.modo)
    raise SystemExit(0 if correctas else 1)


if __name__ == "__main__":
    main()
