"""Entradas de la instancia y resultado esperado (apoyo de E02 y E05).

* La memoria inicial se genera con el procedimiento existente: compilar
  ``operaciones/operaciones.c`` y ejecutar ``./operaciones.out <operacion>``,
  que escribe ``memoria.bin`` (skill 02). No se reescribe en Python.
* Los parámetros de la instancia (tamaños) se leen de los ``#define`` del .c.
  El .c actual tiene tamaños fijos; variar la instancia requiere un .c con
  esos valores (pendiente: parametrización, ver rl/README.md).
* El resultado esperado se calcula en CPU, en Python, a partir de las
  regiones de entrada de esa memoria: es independiente del schedule de los PEs.
"""

import re
import shutil
import subprocess
from pathlib import Path

from ..comun import escribir_json, sha256_archivo, usar_simulador

usar_simulador()
from memoria import ImagenMemoria  # noqa: E402  (módulo del simulador existente)

COMPILADORES = ("clang-18", "clang", "gcc", "cc")

# Regiones de salida de cada operación (las que E05 compara).
REGIONES_SALIDA = {
    "reduccion": ("c", "result"),
    "matmul": ("result",),
    "convolucion": ("result",),
}

# #define relevantes de cada operación en operaciones.c.
PARAMETROS = {
    "reduccion": ("TAMANO_VECTOR",),
    "matmul": ("TAMANO_MATRIZ",),
    "convolucion": ("TAMANO_IMAGEN", "TAMANO_KERNEL"),
}


class ErrorEntradas(RuntimeError):
    """No se pudieron preparar las entradas: problema de integración, no de la acción."""


def parametros_instancia(fuente, operacion):
    texto = Path(fuente).read_text(encoding="utf-8")
    valores = {}
    for nombre in PARAMETROS[operacion]:
        coincidencia = re.search(rf"^#define\s+{nombre}\s+(\d+)", texto, re.M)
        if not coincidencia:
            raise ErrorEntradas(f"{fuente}: no se encontró #define {nombre}")
        valores[nombre] = int(coincidencia.group(1))
    return valores


def generar_memoria(fuente, operacion, carpeta):
    """Compila operaciones.c y ejecuta la operación en ``carpeta``; devuelve memoria.bin."""
    fuente = Path(fuente).resolve()
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    compilador = next((c for c in COMPILADORES if shutil.which(c)), None)
    if compilador is None:
        raise ErrorEntradas(f"No hay compilador de C disponible ({', '.join(COMPILADORES)})")
    ejecutable = carpeta / "operaciones.out"
    compilacion = subprocess.run(
        [compilador, str(fuente), "-I", str(fuente.parent), "-o", str(ejecutable)],
        capture_output=True, text=True,
    )
    if compilacion.returncode != 0:
        raise ErrorEntradas(f"No compiló {fuente}: {compilacion.stderr.strip()}")
    ejecucion = subprocess.run([str(ejecutable), operacion], cwd=carpeta, capture_output=True, text=True)
    ruta = carpeta / "memoria.bin"
    if ejecucion.returncode != 0 or not ruta.is_file():
        raise ErrorEntradas(f"operaciones.out {operacion} falló: {ejecucion.stderr.strip()}")
    return ruta


def _matriz(valores, filas, columnas):
    return [valores[f * columnas:(f + 1) * columnas] for f in range(filas)]


def esperado_reduccion(regiones, formas):
    a, b = regiones["a"], regiones["b"]
    c = [x + y for x, y in zip(a, b)]
    return {"c": c, "result": [sum(c)]}


def esperado_matmul(regiones, formas):
    n, m = formas["a"]
    m2, p = formas["b"]
    if m != m2:
        raise ErrorEntradas(f"matmul: a es {n}x{m} y b es {m2}x{p}")
    a, b = _matriz(regiones["a"], n, m), _matriz(regiones["b"], m2, p)
    c = [sum(a[i][k] * b[k][j] for k in range(m)) for i in range(n) for j in range(p)]
    return {"result": c}


def esperado_convolucion(regiones, formas):
    fi, ci = formas["imagen"]
    fk, ck = formas["kernel"]
    imagen = _matriz(regiones["imagen"], fi, ci)
    kernel = _matriz(regiones["kernel"], fk, ck)
    fs, cs = fi - fk + 1, ci - ck + 1
    salida = [
        sum(imagen[i + ki][j + kj] * kernel[ki][kj] for ki in range(fk) for kj in range(ck))
        for i in range(fs) for j in range(cs)
    ]
    return {"result": salida}


REFERENCIAS = {
    "reduccion": esperado_reduccion,
    "matmul": esperado_matmul,
    "convolucion": esperado_convolucion,
}


def calcular_esperado(ruta_memoria, operacion):
    """Resultado esperado de cada región de salida, calculado desde memoria.bin."""
    regiones, formas = ImagenMemoria.leer(ruta_memoria).regiones()
    esperado = REFERENCIAS[operacion](regiones, formas)
    for nombre, valores in esperado.items():
        if nombre not in formas:
            raise ErrorEntradas(f"La memoria no tiene la región de salida '{nombre}'")
        filas, columnas = formas[nombre]
        if len(valores) != filas * columnas:
            raise ErrorEntradas(f"'{nombre}': esperado {len(valores)} valores, la región tiene {filas * columnas}")
    return {nombre: {"forma": list(formas[nombre]), "valores": valores} for nombre, valores in esperado.items()}


def preparar_instancia(fuente, operacion, carpeta):
    """Genera memoria.bin y esperado.json en ``carpeta`` (artefactos inmutables de la corrida)."""
    carpeta = Path(carpeta)
    ruta_memoria = generar_memoria(fuente, operacion, carpeta)
    (carpeta / "operaciones.out").unlink(missing_ok=True)
    esperado = calcular_esperado(ruta_memoria, operacion)
    _, formas = ImagenMemoria.leer(ruta_memoria).regiones()
    instancia = {
        "regiones_memoria": {nombre: list(forma) for nombre, forma in formas.items()},
        "operacion": operacion,
        "parametros_instancia": parametros_instancia(fuente, operacion),
        "fuente_c": str(fuente),
        "hash_fuente_c": sha256_archivo(fuente),
        "memoria_inicial": str(ruta_memoria),
        "hash_memoria_inicial": sha256_archivo(ruta_memoria),
        "regiones_salida": list(REGIONES_SALIDA[operacion]),
        "esperado": str(carpeta / "esperado.json"),
    }
    escribir_json(carpeta / "esperado.json", {"hash_memoria_inicial": instancia["hash_memoria_inicial"],
                                              "regiones": esperado})
    escribir_json(carpeta / "instancia.json", instancia)
    return instancia
