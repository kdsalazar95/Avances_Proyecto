# Plan de pruebas

## Descripción pruebas:

## Nivel HW — Medir el circuito real: compuertas, conexiones, velocidad, recursos

Se evalua qué tan caro y rápido es el circuito una vez convertido a hardware real, usando Vitis.

_Los datos salen del reporte de síntesis de Vitis HLS (por ejemplo:`csynth.rpt`), y del reporte posterior a la implementación cuando aplique._


| ID | Qué se mide | Cómo se mide | Qué se espera ver |
|---|---|---|---|
| TC-H1.1 | Cuántas compuertas usa cada componente del PE por separado | Sintetizar por separado la Functional Unit, el Register File y el router del NoC en Vitis HLS | Un número de compuertas por componente, que sirva para comparar cuál pieza es "más cara" |
| TC-H1.2 | Cuántas compuertas usa la malla completa | Sintetizar en Vitis HLS toda la configuración con el acomodo de celdas propuesto | El total debe ser la suma de las celdas usadas más el circuito de conexión, sin que falte ni sobre nada |
| TC-H1.3 | Cuántas conexiones (cables) se usan de verdad | Revisar en el reporte cuántos enlaces están realmente activos según cómo se conectaron las celdas | El número debe coincidir con las conexiones que realmente se usaron, ni una más ni una menos |
| TC-H1.4 | Cuánto tarda una operación en completarse | Correr la simulación/síntesis en Vitis y leer cuántos ciclos (o nanosegundos) tarda una operación | El tiempo debe ser mayor si el dato tiene que pasar por más celdas en el camino |
| TC-H1.5 | A qué velocidad máxima puede correr el circuito (frecuencia) | Leer la frecuencia máxima (Fmax) del reporte de Vitis | Esa velocidad máxima debe bajar si las conexiones son más largas o hay más pasos entre celdas |
| TC-H1.6 | Cuántos recursos del chip se usan (LUT, FF, DSP, BRAM) | Leer esos números del reporte de Vitis para una configuración dada | Deben corresponder exactamente al tipo y cantidad de celdas que se colocaron |
| TC-H1.7 | Qué porcentaje de los recursos disponibles se está usando | Calcular recursos usados entre recursos disponibles | Debe reflejar bien cuántas celdas están realmente trabajando |
| TC-H1.8 | Cuántas operaciones se completan por segundo (throughput) | Calcular usando la velocidad (Fmax) y cuántos ciclos tarda cada operación | Debe cuadrar con los otros dos datos: si la velocidad y el tiempo por operación no cambiaron, el throughput tampoco debería cambiar |
| TC-H1.9 | Que los resultados sean siempre iguales al repetir la medición | Sintetizar la misma configuración dos veces | Las compuertas, conexiones, recursos y velocidad deben salir exactamente iguales las dos veces |
| TC-H1.10 | Que el acomodo de las celdas sí afecte el resultado | Comparar dos configuraciones con las mismas celdas, pero acomodadas distinto en la malla | Las conexiones, el tiempo y la velocidad deben cambiar según qué tan cerca o lejos están las celdas que se necesitan entre sí |

## Nivel SW — Qué tan bien aprende el agente (software)

Se mide donde el agente decide poner cada celda y cómo conectarlas.

| ID | Qué se prueba | Cómo se prueba | Qué se espera ver |
|---|---|---|---|
| TC-RL2.1 | El agente va mejorando con el tiempo | Entrenarlo durante muchas rondas (episodios) sobre el mismo problema | La recompensa promedio debe ir subiendo de forma constante, no quedarse igual ni saltar sin control |
| TC-RL2.2 | El agente aprende algo confiable, no por suerte | Entrenarlo varias veces, cambiando el punto de partida aleatorio cada vez | Los resultados finales deben ser parecidos entre una corrida y otra |
| TC-RL2.3 | El agente realmente hace un mejor trabajo que hacerlo a mano o al azar | Comparar la configuración que arma el agente contra una hecha a mano o elegida al azar | El agente debe igualar o superar esa comparación en al menos una de las métricas del Nivel 1 |
| TC-RL2.4 | El agente sabe balancear varios objetivos que compiten entre sí | Revisar casos donde usar menos compuertas implica más tiempo de operación, o viceversa | El agente debe encontrar un balance razonable entre ambos, no ignorar uno por completo |
| TC-RL2.5 | Lo que se evalúa es exactamente lo que se usa para corregir al agente | Revisar que la configuración que se evaluó sea la misma que se usa para "reacomodar" las celdas después | No debe haber diferencia entre lo evaluado y lo que realmente se ajusta |
| TC-RL2.6 | El agente se adapta a mallas de distinto tamaño | Entrenarlo/probarlo en una malla chica y en una más grande | Debe encontrar una buena solución en ambos casos, sin fallar |
| TC-RL2.7 | El agente se adapta a distintos tipos de tareas (kernels) | Entrenarlo/probarlo con tareas distintas, como multiplicación de matrices y una suma acumulada | Debe encontrar acomodos distintos y apropiados para cada tarea |
| TC-RL2.8 | El agente nunca propone algo imposible de construir | Revisar cada configuración que propone antes de evaluarla | Ninguna debe tener celdas fuera de la malla ni conexiones que no existen |

---

# Pruebas a nivel de código 

Convertir el plan de pruebas en código implica construir **dos motores completamente distintos**, porque HW y SW dependen de herramientas y flujos muy diferentes. Este documento desglosa qué necesita cada uno.

## Índice

- [Nivel HW — automatizar Vitis HLS y parsear reportes](#nivel-hw--automatizar-vitis-hls-y-parsear-reportes)
  - [1. Script de síntesis parametrizable (TCL)](#1-script-de-síntesis-parametrizable-tcl)
  - [2. Parser de reportes (Python)](#2-parser-de-reportes-python)
  - [3. Test runner (pytest)](#3-test-runner-pytest)
  - [Nota sobre "compuertas" y "conexiones"](#nota-sobre-compuertas-y-conexiones)
- [Nivel SW — el entorno de RL y el bucle de entrenamiento](#nivel-sw--el-entorno-de-rl-y-el-bucle-de-entrenamiento)
  - [1. El entorno (environment)](#1-el-entorno-environment)
  - [2. El cuello de botella: velocidad de evaluación](#2-el-cuello-de-botella-velocidad-de-evaluación)
  - [3. Función de recompensa multiobjetivo](#3-función-de-recompensa-multiobjetivo)
  - [4. Bucle de entrenamiento y logging](#4-bucle-de-entrenamiento-y-logging)
  - [5. Comparación contra baseline](#5-comparación-contra-baseline)
- [Decisiones pendientes antes de escribir código real](#decisiones-pendientes-antes-de-escribir-código-real)

---

## Nivel HW — automatizar Vitis HLS y parsear reportes

Se necesitan tres piezas de código, encadenadas: un script de síntesis, un parser de resultados, y un runner de pruebas que los orquesta.

### 1. Script de síntesis parametrizable (TCL)

Vitis HLS se controla por línea de comandos con TCL. Se necesita un script que reciba la configuración (placement/routing) como parámetro y sintetice ese diseño específico.

```tcl
# synth_config.tcl
open_project -reset cgra_proj
set_top cgra_top
add_files src/cgra_top.cpp
add_files -tb src/testbench.cpp
open_solution "solution1"
set_part {xck26-sfvc784-2LV-c}
create_clock -period 4 -name default   ;# 250 MHz, ajustable

csynth_design   ;# genera csynth.rpt

exit
```

Este script corre con:

```bash
vitis_hls -f synth_config.tcl
```

y genera el reporte en `solution1/syn/report/csynth.rpt`.

### 2. Parser de reportes (Python)

`csynth.rpt` es texto estructurado — se necesita extraer LUT/FF/DSP/BRAM, latencia y ciclos con expresiones regulares o parseo de secciones.

```python
import re
from pathlib import Path

def parse_csynth_report(path: str) -> dict:
    text = Path(path).read_text()

    def grab(pattern, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "lut": grab(r"LUT\s*\|\s*(\d+)"),
        "ff": grab(r"FF\s*\|\s*(\d+)"),
        "dsp": grab(r"DSP\s*\|\s*(\d+)"),
        "bram": grab(r"BRAM_18K\s*\|\s*(\d+)"),
        "latency_cycles": grab(r"Latency \(cycles\)\D*(\d+)"),
        "fmax_mhz": grab(r"Estimated Frequency\D*([\d.]+)", cast=float),
    }
```

> **Importante**: `csynth.rpt` (síntesis lógica de HLS) da recursos **estimados**, no el número real tras place & route. Para Fmax real (`TC-H1.5`) y, en algunos casos, recursos reales (`TC-H1.6`), como vimos en el paper (Tabla III dice "VIVADO"), también hay que parsear el reporte de implementación de **Vivado**, con la misma lógica pero apuntando a otro archivo (`impl/report/verilog/...`).

### 3. Test runner (pytest)

```python
import subprocess
import pytest

def run_synthesis(config_path: str) -> dict:
    subprocess.run(
        ["vitis_hls", "-f", "synth_config.tcl", "-tclargs", config_path],
        check=True,
    )
    return parse_csynth_report("solution1/syn/report/csynth.rpt")


def test_TC_H1_9_reproducibilidad():
    r1 = run_synthesis("configs/config_a.json")
    r2 = run_synthesis("configs/config_a.json")
    assert r1 == r2


def test_TC_H1_10_sensibilidad_placement():
    r_cerca = run_synthesis("configs/celdas_juntas.json")
    r_lejos = run_synthesis("configs/celdas_separadas.json")
    assert r_lejos["latency_cycles"] > r_cerca["latency_cycles"]
```

### Nota sobre "compuertas" y "conexiones"

Vitis HLS no reporta "número de compuertas" ni "número de conexiones" directamente como cifras nativas:

- Reporta **LUT/FF/DSP**, que en flujo FPGA son el equivalente práctico a "compuertas".
- El conteo de **interconexiones** normalmente sale del netlist post-síntesis (`.v`/`.vhd` generado) o del reporte de utilización de rutas de Vivado.

Vale la pena confirmar con el profesor si "compuertas" significa LUT/FF (lo estándar en flujo FPGA) o un conteo de compuertas lógicas equivalentes estilo ASIC (que normalmente no sale de Vitis, sino de una herramienta de síntesis ASIC como Design Compiler).

---

## Nivel SW — el entorno de RL y el bucle de entrenamiento

Aquí el problema central es distinto: hay que modelar el placement + routing como un **problema de decisión** (entorno tipo Gym) y un agente que interactúe con él.

### 1. El entorno (environment)

```python
import gymnasium as gym
import numpy as np

class CGRAPlacementEnv(gym.Env):
    def __init__(self, rows: int, cols: int, kernel: str):
        self.rows, self.cols = rows, cols
        self.kernel = kernel
        # acción: dónde colocar la siguiente celda / cómo enrutarla
        self.action_space = gym.spaces.Discrete(rows * cols)
        # estado: el grafo actual (qué celda hay en cada posición)
        self.observation_space = gym.spaces.Box(0, 1, shape=(rows, cols))

    def reset(self, seed=None):
        self.grafo = np.zeros((self.rows, self.cols))
        return self.grafo, {}

    def step(self, action):
        self._aplicar_placement(action)          # TC-RL2.8: validar aquí
        config = self._grafo_a_config()
        metrics = self._evaluar(config)           # Nivel HW: rápido o real
        reward = self._calcular_recompensa(metrics)  # TC-RL2.4
        done = self._config_completa()
        return self.grafo, reward, done, False, metrics
```

### 2. El cuello de botella: velocidad de evaluación

Una síntesis real en Vitis HLS toma minutos; un entrenamiento de RL típico necesita miles de evaluaciones por episodio. **No es viable llamar a Vitis en cada `step()`.**

Se necesita separar dos etapas:

| Etapa | Qué usar | Por qué |
|---|---|---|
| Durante entrenamiento | Un estimador rápido (`AreaModel`/`PowerAnalyzer` de `ppa_engine.py`, o uno nuevo calibrado con corridas reales) | El agente necesita explorar miles de configuraciones por hora |
| Validación periódica | Síntesis real en Vitis, cada N episodios o al final | Confirma que el estimador no se desvió — es lo que valida `TC-H1.9`/`TC-H1.10` aplicado como chequeo de calibración |

### 3. Función de recompensa multiobjetivo

Corresponde a `TC-RL2.4`.

```python
def _calcular_recompensa(self, metrics: dict) -> float:
    w_compuertas, w_latencia, w_fmax = 0.4, 0.4, 0.2
    return -(
        w_compuertas * metrics["lut"] / LUT_MAX
        + w_latencia * metrics["latency_cycles"] / LAT_MAX
        - w_fmax * metrics["fmax_mhz"] / FMAX_MAX
    )
```

### 4. Bucle de entrenamiento y logging

Corresponde a `TC-RL2.1` y `TC-RL2.2`.

```python
from stable_baselines3 import PPO

def entrenar(seed: int, episodios: int):
    env = CGRAPlacementEnv(rows=4, cols=4, kernel="gemm")
    modelo = PPO("MlpPolicy", env, seed=seed, verbose=1)
    modelo.learn(total_timesteps=episodios * 100)
    return modelo, env.historial_recompensas   # para graficar TC-RL2.1
```

### 5. Comparación contra baseline

Corresponde a `TC-RL2.3`.

```python
def placement_aleatorio(env) -> dict:
    ...  # coloca celdas al azar, sin aprendizaje

def comparar_contra_baseline(modelo_rl, env, n_pruebas=20):
    resultados_rl = [evaluar(modelo_rl, env) for _ in range(n_pruebas)]
    resultados_random = [placement_aleatorio(env) for _ in range(n_pruebas)]
    assert np.mean([r["lut"] for r in resultados_rl]) <= np.mean([r["lut"] for r in resultados_random])
```

---

## Decisiones pendientes antes de escribir código real

1. **¿Ya existe el diseño HLS parametrizable** (el `.cpp`/SystemC que Vitis puede sintetizar variando placement/routing por parámetro), o también hay que construirlo?
2. **¿Qué librería de RL se va a usar** — `stable-baselines3` (más simple, algoritmos estándar como PPO/DQN) o algo hecho a mano (más control, más trabajo)?
3. **¿Ya existe un estimador rápido de métricas** para entrenamiento, o hay que calibrarlo desde cero con las primeras corridas reales de Vitis?