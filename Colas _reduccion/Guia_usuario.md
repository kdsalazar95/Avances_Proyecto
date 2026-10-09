# Guia de uso

## Requerimientos

| Entorno | Herramienta |
|---|---|
| Python | UV |
| Compilacion e IR | `clang-18` |
| Graficas LLVM | `opt-18` |

Desde la carpeta del proyecto, sincroniza el entorno:

```bash
uv sync
uv venv
```

El proyecto usa `uv run`, por lo que no es necesario activar manualmente el
entorno virtual. Verifica que tu sistema tenga disponibles `python3`,
`clang-18` y `opt-18`.

## Fuentes permitidas

El flujo de skills usa solamente estas fuentes C:

| Operacion | Fuente | Parametros |
|---|---|---|
| Reduccion | `src/reduccion/reduccion.c` | `TAMANO_VECTOR` |
| Multiplicacion de matrices | `src/matmul/matmul.c` | `TAMANO_MATRIZ` |
| Convolucion | `src/convolucion/convolucion.c` | `TAMANO_IMAGEN`, `TAMANO_KERNEL` |

Los tamanos se cambian en los `#define` al inicio de cada fuente. Al cambiar un
tamano hay que volver a ejecutar los skills, porque el horario de los PEs lleva
los indices de memoria ya calculados: si no, la CGRA falla con un indice fuera
de rango. Los skills tambien vuelven a generar `memoria.bin` (ver
[Memoria](#memoria)).

## Configuracion de la CGRA

La CGRA se describe en `src/cgra.json`. Lo leen tanto `run_cgra.py` como los
skills, asi que el compilador y el simulador siempre usan la misma CGRA:

```json
{
  "filas": 4,
  "columnas": 4,
  "palabras_memoria": 4096,
  "tipos_pe": {
    "M": {"nombre": "memoria", "instrucciones": ["NOP", "MOV", "LD", "ST", "SEND", "RECV"]},
    "C": {"nombre": "computo", "instrucciones": ["NOP", "MOV", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV"]}
  },
  "distribucion": {"oeste": "M", "este": "M", "resto": "C"}
}
```

- `filas x columnas` es la cantidad de PEs.
- `palabras_memoria` es el tamano de la memoria de datos.
- `tipos_pe` dice que instrucciones puede ejecutar cada tipo.
- `distribucion` es la regla que da el tipo de cada PE: los de la columna
  oeste y este son de memoria (`M`) y el resto de computo (`C`). Los bordes
  posibles son `norte`, `sur`, `oeste` y `este`; en una esquina mandan
  `oeste`/`este`.

Para cambiar el tamano basta con cambiar `filas` y `columnas`: la malla se
arma sola con la regla. Por ejemplo, con `"filas": 8, "columnas": 8` queda:

```text
M C C C C C C M
M C C C C C C M
...              (8 filas)
```

Para una distribucion especial que no sigue la regla, se puede escribir la
malla completa en `"malla"` (una fila por linea, por ejemplo `"M C C M"`), que
reemplaza a `distribucion`.

### Tipos de PE

| Tipo | Donde | Puede | No puede |
|---|---|---|---|
| **memoria (M)** | columnas oeste y este | `LD`, `ST`, `SEND`, `RECV`, `MOV`, `NOP` | calcular |
| **computo (C)** | centro | `ADD`, `SUB`, `MUL`, `DIV`, `SEND`, `RECV`, `MOV`, `NOP` | acceder a memoria |

Los PEs de memoria leen los operandos y se los envian a los PEs de computo;
estos calculan y devuelven el resultado para que el PE de memoria lo guarde.
Si un programa usa una instruccion que el tipo de su PE no permite,
`run_cgra.py` no lo carga e indica el PE y el ciclo:

```text
No se pudo cargar el programa: 944 instrucciones no respetan el tipo de su PE:
  PE00 (memoria), ciclo 30: ADD no está permitido
  ...
```

Para cambiar la CGRA (otro tamano u otra distribucion de tipos) se edita
`src/cgra.json` y se vuelven a correr los skills. Para probar otro formato sin
tocar el de por defecto:

```bash
uv run src/run_cgra.py src/matmul --cgra otra_cgra.json
```

La CGRA funciona como un procesador: **no se le indica que operacion va a
ejecutar**. Cada PE es un procesador pequeno que ejecuta su programa
(`PE{fila}{columna}.txt`) sobre la memoria (`memoria.bin`), y ambos los
producen los skills a partir del `.c`. La operacion solo se elige al pedir que
se corran los skills, indicando la fuente (por ejemplo, "corre los skills para
`src/matmul/matmul.c`").

La malla y el programa no tienen que ser del mismo tamano:

- **Programa mayor que la malla:** se procesa por tiles del tamano de la malla,
  uno tras otro (ver [Operaciones por tiles](#operaciones-por-tiles)).
- **Malla mayor que el programa:** los PEs que sobran quedan en `NOP`. En una
  reduccion arrancan igual con `MOV acc, 0.0`, para no alterar la ruta.

## Preparar el pipeline de skills

Los skills se ejecutan en este orden:

1. Recibir la fuente indicada y leer la malla de `run_cgra.py`.
2. Generar el archivo LLVM con `clang-18`, y compilar y ejecutar el `.c` para
   generar la memoria inicial `memoria.bin`.
3. Generar las graficas CFG y DDG con `opt-18`.
4. Analizar el programa, detectar paralelismo y mapearlo a la malla.
5. Calendarizar las instrucciones.
6. Generar y validar un archivo por PE.

Todo lo que genera el pipeline queda en la carpeta del programa:

```text
src/<programa>/
├── <programa>.c             fuente, con sus #define e inicializar_memoria()
├── <programa>.ll            skill 02
├── memoria.bin              skill 02, memoria inicial que escribe el .c
├── .main.dot                skill 03 (CFG)
├── ddg.main..dot            skill 03 (DDG)
├── pe_instructions/         skill 08
│   └── PE{fila}{columna}.txt
└── reporte_ciclos.txt       reporte de la ultima corrida
```

Cada programa tiene su carpeta, asi que compilar uno no afecta a los horarios
de los demas. Al regenerar con la misma malla, los archivos se
sobrescriben. **Si cambias el tamano de la malla, borra antes los `PE*.txt`
anteriores:** con 10 o mas filas o columnas cada coordenada lleva dos digitos
(`PE0100.txt`), asi que los nombres nuevos no pisan a los viejos y quedan los
dos juegos mezclados en la carpeta.

`test/datos/reduccion_2x2` no forma parte de esa salida: es el horario de
referencia `2x2` de la reduccion, usado por las pruebas.

Los detalles de cada paso estan en `agent_skills/README.md`.

## Operaciones por tiles

La salida de `matmul` o de `convolucion` puede ser mayor que la malla. En ese
caso se procesa por tiles. Solo los PEs de computo calculan, asi que el tile es
el bloque de PEs de computo: `4x2` en la CGRA por defecto. Una salida `9x9`
necesita `3 x 5 = 15` tiles.

Cada PE de computo calcula un elemento del tile. El PE de memoria de su misma
fila (el vecino oeste para la columna 1, el vecino este para la columna 2) le
carga los dos operandos de cada paso con `LD` y se los envia con `SEND`. Al
terminar el tile, el PE de computo devuelve su resultado y el PE de memoria lo
guarda con `ST`. En un tile de borde, los PEs que quedan fuera hacen `NOP`.

En una reduccion el tiling es distinto: cada PE de computo acumula localmente su
parte del vector, despues se reduce entre los PEs de computo hacia `PE01`, y
`PE01` le manda el total a `PE00`, que lo guarda en `result[0]`.

## Ejecutar la CGRA

`run_cgra.py` recibe la carpeta del programa, igual que a un procesador se le
da un ejecutable, y carga de ahi `pe_instructions/` y `memoria.bin`:

```bash
uv run src/run_cgra.py src/convolucion
```

No hay que indicar la operacion ni el tamano del resultado: al terminar se
muestran las regiones de memoria que escribieron los PEs, cada una con la forma
que declara `memoria.bin` (un valor, un vector o una matriz).

Para correr con una CGRA distinta de la configurada:

```bash
uv run src/run_cgra.py src/reduccion --cgra otra_cgra.json
```

Los `PE*.txt` de la carpeta tienen que corresponder a esa CGRA (tamano y tipos).

### Reporte de ciclos

Cada corrida guarda un reporte en `src/<programa>/reporte_ciclos.txt`, con un
solo nombre por programa que se reescribe en la corrida siguiente. Contiene:

- La malla, el programa y las regiones de su memoria.
- Cuantos ciclos se van en computo, en comunicacion y cuantos quedan inactivos.
- Cada desplazamiento de datos entre PEs: ciclo, emisor, receptor y direccion.

Con `--ciclos` se muestra ademas en pantalla, y con `--reporte-ciclos RUTA` se
guarda en otra ruta. En `matmul` y `convolucion` no hay comunicacion entre PEs,
asi que el reporte lo dice explicitamente.

### Memoria

La memoria de la CGRA es binaria, como la de un procesador, y no la arma
Python: la genera el propio `.c`. Cada fuente tiene una funcion
`inicializar_memoria()` que crea e inicializa los arreglos y al final los guarda
en `memoria.bin` con `guardar_memoria()` de `src/memoria_cgra.h`. El skill 02
compila y ejecuta el `.c` para producirla; a mano seria:

```bash
cd src/matmul
clang-18 matmul.c -o matmul.out && ./matmul.out && rm matmul.out
```

El archivo es una memoria plana de `float32` con una tabla de simbolos al
inicio: el nombre de cada region, su primera palabra y su forma. El formato
exacto esta documentado en `src/memoria_cgra.h`.

| Programa | Regiones |
|---|---|
| Reduccion | `a`, `b`, `c`, `result` |
| Matmul | `a`, `b`, `result` (la matriz C) |
| Convolucion | `imagen`, `kernel`, `result` |

Las matrices se guardan aplanadas fila por fila. Los programas de los PEs usan
estos nombres de region en sus `LD`/`ST`.

`memoria.bin` es una imagen, no la memoria de la CGRA. Al correr un programa,
`run_cgra.py` hace lo mismo que un sistema operativo con un ejecutable:

1. **Enciende** la CGRA (`src/cgra.py`) segun `src/cgra.json`: una memoria de
   datos de `palabras_memoria` palabras, toda en cero, y PEs con su tipo,
   registros en cero y sin programa.
2. **Carga** el programa (`src/cargador.py`): verifica que la imagen quepa,
   que cada PE solo use instrucciones de su tipo, la copia al inicio de la
   memoria (el resto sigue en cero), convierte cada
   `a[3]` de los `LD`/`ST` en la direccion de esa palabra y le da a cada PE su
   programa.
3. **Ejecuta** los ciclos.

Si la imagen no cabe en la memoria, la carga falla con un mensaje en vez de
cortarla; en ese caso hay que aumentar `palabras_memoria` en `src/cgra.json`.

Para usar otra memoria u otros programas de PE:

```bash
uv run src/run_cgra.py src/matmul --memoria otra_memoria.bin
uv run src/run_cgra.py src/matmul --instrucciones otra_carpeta/
```

### Opciones utiles

```bash
# Mostrar en pantalla los ciclos de computo y comunicacion
uv run src/run_cgra.py src/convolucion --ciclos

# Mostrar la traza: en cada ciclo, que instruccion ejecuto cada PE (sin los
# NOP) y que escribio en registros o memoria. Sirve para cualquier programa.
uv run src/run_cgra.py src/reduccion --traza

# Guardar el reporte en otra ruta
uv run src/run_cgra.py src/convolucion \
	--reporte-ciclos programas/reportes_ciclos/reporte.txt

# Guardar la memoria resultante, con los resultados, en el memoria.bin usado
uv run src/run_cgra.py src/convolucion --write-back
```

## Archivos generados

Lo que producen los skills y `run_cgra.py` no se versiona: `.gitignore` excluye
los `.ll`, los `.dot`, las carpetas `pe_instructions/`, los `memoria.bin` y los
ejecutables `.out`.
Todo eso se vuelve a crear ejecutando el pipeline y la CGRA.

## Pruebas

```bash
uv run test/run_tests.py
```
