# De plan de pruebas a código

El plan de pruebas en código implica construir **dos motores completamente distintos**, porque HW y SW dependen de herramientas y flujos muy diferentes. A continuación se desglosa qué necesita cada uno.

## Índice

- [Niveles 0 a 4 — pruebas de correctitud del simulador (pytest)](#niveles-0-a-4--pruebas-de-correctitud-del-simulador-pytest)
  - [Nivel 0 — el NoC solo](#nivel-0--el-noc-solo)
  - [Nivel 1 — un dato viajando entre vecinos](#nivel-1--un-dato-viajando-entre-vecinos)
  - [Nivel 2 — cada PE recibe su instrucción](#nivel-2--cada-pe-recibe-su-instrucción)
  - [Nivel 3 — la malla completa, varios ciclos](#nivel-3--la-malla-completa-varios-ciclos)
  - [Nivel 4 — comparación contra el modelo de referencia](#nivel-4--comparación-contra-el-modelo-de-referencia)
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

## Niveles 0 a 4 — pruebas de correctitud del simulador (pytest)

Estas pruebas no necesitan Vitis ni ninguna herramienta externa: corren directamente sobre `archpilot_cgra`, con `pytest`. Son el prerrequisito de todo lo demás — si algo falla aquí, ni las métricas de hardware ni el entrenamiento del agente son confiables.

### Nivel 0 — el NoC solo

```python
import pytest
from archpilot_cgra.noc import NoC
from archpilot_cgra.exceptions import SimulationException


def test_TC_N0_1_vecinos_interior_mesh():
    noc = NoC(rows=3, cols=3, topology="mesh")
    vecinos = noc.get_neighbors(1, 1)
    assert vecinos["north"] == (0, 1)
    assert vecinos["south"] == (2, 1)
    assert vecinos["east"] == (1, 2)
    assert vecinos["west"] == (1, 0)


def test_TC_N0_2_vecinos_esquina_mesh():
    noc = NoC(rows=3, cols=3, topology="mesh")
    vecinos = noc.get_neighbors(0, 0)
    assert vecinos["north"] is None
    assert vecinos["west"] is None
    assert vecinos["south"] == (1, 0)
    assert vecinos["east"] == (0, 1)


def test_TC_N0_3_vecinos_esquina_torus():
    noc = NoC(rows=3, cols=3, topology="torus")
    vecinos = noc.get_neighbors(0, 0)
    assert vecinos["north"] == (2, 0)
    assert vecinos["west"] == (0, 2)
    assert None not in vecinos.values()


def test_TC_N0_4_topologia_bloqueada_durante_ejecucion():
    noc = NoC(rows=2, cols=2, topology="mesh")
    noc.lock()
    with pytest.raises(SimulationException):
        noc.set_topology("torus")


def test_TC_N0_5_configuraciones_invalidas():
    with pytest.raises(SimulationException):
        NoC(rows=0, cols=2)
    with pytest.raises(SimulationException):
        NoC(rows=2, cols=2, topology="ring")
```

### Nivel 1 — un dato viajando entre vecinos

```python
from archpilot_cgra import ConfigMemory, PEArray


def test_TC_N1_1_snapshot_no_valor_a_medio_calcular():
    mem = ConfigMemory(rows=1, cols=2, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
    ])
    array = PEArray(rows=1, cols=2, config_memory=mem)
    array.get_pe(0, 0).rf.write(0, 10)
    array.get_pe(0, 0).rf.write(1, 5)

    array.step()  # ciclo 0: PE(0,0) calcula 15, pero el vecino aún no lo ve
    assert array.get_pe(0, 1).neighbor_inputs["west"] == 0

    array.step()  # ciclo 1: ahora sí llega el valor del ciclo anterior
    assert array.get_pe(0, 1).neighbor_inputs["west"] == 15


def test_TC_N1_2_borde_recibe_cero():
    array = PEArray(rows=1, cols=1, topology="mesh")
    array.step()
    assert array.get_pe(0, 0).neighbor_inputs["north"] == 0
    assert array.get_pe(0, 0).neighbor_inputs["west"] == 0


def test_TC_N1_3_lee_vecino_correcto():
    mem = ConfigMemory(rows=1, cols=2, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 2, "mux_sel": 0}},
        {(0, 1): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 2, "mux_sel": 4}},  # 4 = oeste
    ])
    array = PEArray(rows=1, cols=2, config_memory=mem)
    array.get_pe(0, 0).rf.write(0, 42)
    array.run(2)
    assert array.get_pe(0, 1).neighbor_inputs["west"] == 42


def test_TC_N1_4_multicast_mismo_vecino():
    mem = ConfigMemory(rows=1, cols=3, config_sequence=[
        {(0, 1): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 1, "mux_sel": 0}},
    ])
    array = PEArray(rows=1, cols=3, config_memory=mem)
    array.get_pe(0, 1).rf.write(0, 99)
    array.run(2)
    assert array.get_pe(0, 0).neighbor_inputs["east"] == 99
    assert array.get_pe(0, 2).neighbor_inputs["west"] == 99
```

### Nivel 2 — cada PE recibe su instrucción

```python
import pytest
from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.exceptions import ConfigMemoryException, SimulationException


def test_TC_P2_1_cada_pe_ejecuta_su_instruccion():
    mem = ConfigMemory(rows=1, cols=2, config_sequence=[
        {
            (0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0},
            (0, 1): {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0},
        }
    ])
    array = PEArray(rows=1, cols=2, config_memory=mem)
    array.get_pe(0, 0).rf.write(0, 3)
    array.get_pe(0, 0).rf.write(1, 4)
    array.get_pe(0, 1).rf.write(0, 3)
    array.get_pe(0, 1).rf.write(1, 4)

    array.step()

    assert array.get_pe(0, 0).rf.read(2) == 7    # ADD
    assert array.get_pe(0, 1).rf.read(2) == 12   # MUL


def test_TC_P2_2_pe_sin_instruccion_no_hace_nada():
    mem = ConfigMemory(rows=1, cols=2, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
    ])
    array = PEArray(rows=1, cols=2, config_memory=mem)
    array.step()
    assert array.get_pe(0, 1).fu.last_opcode == "NOP"


def test_TC_P2_3_pe_fuera_de_malla():
    with pytest.raises(ConfigMemoryException):
        ConfigMemory(rows=1, cols=1, config_sequence=[
            {(5, 5): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
        ])


def test_TC_P2_4_mux_sel_invalido():
    mem = ConfigMemory(rows=1, cols=1, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 9}},
    ])
    array = PEArray(rows=1, cols=1, config_memory=mem)
    with pytest.raises(SimulationException):
        array.step()
```

### Nivel 3 — la malla completa, varios ciclos

```python
from archpilot_cgra import ConfigMemory, PEArray


def test_TC_M3_1_multihop_un_ciclo():
    mem = ConfigMemory(rows=1, cols=2, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 2, "mux_sel": 0}},
        {(0, 1): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 2, "mux_sel": 4}},
    ])
    array = PEArray(rows=1, cols=2, config_memory=mem)
    array.get_pe(0, 0).rf.write(0, 21)
    historial = array.run(2)
    assert historial[1]["results"][(0, 1)] == 21


def test_TC_M3_2_mismo_programa_2x2_y_4x4():
    def correr(rows, cols):
        mem = ConfigMemory(rows=rows, cols=cols, config_sequence=[
            {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
        ])
        array = PEArray(rows=rows, cols=cols, config_memory=mem)
        array.get_pe(0, 0).rf.write(0, 10)
        array.get_pe(0, 0).rf.write(1, 5)
        array.run(1)
        return array.get_pe(0, 0).rf.read(2)

    assert correr(2, 2) == correr(4, 4) == 15


def test_TC_M3_3_mesh_vs_torus_bordes():
    def correr(topology):
        mem = ConfigMemory(rows=2, cols=2, config_sequence=[
            {(1, 0): {"opcode": "ADD", "src_a": 0, "src_b": 0, "dst": 1, "mux_sel": 0}},
        ])
        array = PEArray(rows=2, cols=2, topology=topology, config_memory=mem)
        array.get_pe(1, 0).rf.write(0, 77)
        array.run(2)
        return array.get_pe(0, 0).neighbor_inputs["north"]

    assert correr("mesh") == 0     # (0,0) no tiene vecino norte en mesh
    assert correr("torus") == 77   # en torus, el norte de (0,0) es (1,0)


def test_TC_M3_4_reset_es_reproducible():
    mem = ConfigMemory(rows=1, cols=1, config_sequence=[
        {(0, 0): {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
    ])
    array = PEArray(rows=1, cols=1, config_memory=mem)

    array.get_pe(0, 0).rf.write(0, 6)
    array.get_pe(0, 0).rf.write(1, 7)
    r1 = array.run(1)[0]["results"][(0, 0)]

    array.reset()
    array.get_pe(0, 0).rf.write(0, 6)
    array.get_pe(0, 0).rf.write(1, 7)
    r2 = array.run(1)[0]["results"][(0, 0)]

    assert r1 == r2 == 42
```

### Nivel 4 — comparación contra el modelo de referencia

```python
from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.golden_model import GoldenModel


def test_TC_G4_1_gemm_correcto_end_to_end():
    from benchmarks.gemm_benchmark import run_default_gemm
    resultado = run_default_gemm()
    assert resultado.validation.passed is True


def test_TC_G4_2_resultados_intermedios():
    mem = ConfigMemory(rows=1, cols=1, config_sequence=[
        {(0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}},
        {(0, 0): {"opcode": "MUL", "src_a": 2, "src_b": 3, "dst": 0, "mux_sel": 0}},
    ])
    array = PEArray(rows=1, cols=1, config_memory=mem)
    array.get_pe(0, 0).rf.write(0, 10)
    array.get_pe(0, 0).rf.write(1, 5)
    array.get_pe(0, 0).rf.write(3, 3)

    historial = array.run(2)

    assert historial[0]["results"][(0, 0)] == GoldenModel.add(10, 5)
    assert historial[1]["results"][(0, 0)] == GoldenModel.multiply(15, 3)
```

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

>__Importante:__ `csynth.rpt` (síntesis lógica de HLS) da recursos **estimados**, no el número real tras place & route. Para Fmax real (`TC-H1.5`) y, en algunos casos, recursos reales (`TC-H1.6`), como vimos en el paper (Tabla III dice "VIVADO"), también hay que parsear el reporte de implementación de **Vivado**, con la misma lógica pero apuntando a otro archivo (`impl/report/verilog/...`).

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