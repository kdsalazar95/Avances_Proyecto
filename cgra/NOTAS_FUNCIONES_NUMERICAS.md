# Funciones numéricas retiradas

`formatear_numero()` está activa para mostrar los números con una representación
compacta. La normalización de floats enteros sigue documentada como una opción
retirada; los valores de memoria y las constantes `MOV` se conservan como
floats.

## `formatear_numero(valor)`

```python
def formatear_numero(valor):
    return f"{round(valor, 6):g}" if isinstance(valor, (int, float)) else str(valor)
```

Convierte un valor a texto compacto: por ejemplo, `2.0` pasa a `"2"` y
`1.23456789` pasa a `"1.23457"`. Solo cambia la representación mostrada, no el
valor almacenado.

Está definida en `src/formato_numero.py` y se usa desde:

- `src/consola.py`, al mostrar listas y vectores.
- `src/isa.py`, al volver a escribir una instrucción `MOV` como texto.
- `src/resultados.py`, al mostrar un resultado escalar.
- `src/traza.py`, al mostrar los efectos de instrucciones.

## `normalizar_numero(valor)`

```python
def normalizar_numero(valor):
    return int(valor) if isinstance(valor, float) and valor.is_integer() else valor
```

Convierte un float que representa un entero, como `2.0`, al entero `2`. No
redondea valores como `2.3`.

Antes, este comportamiento estaba implementado por separado en:

- `src/memoria.py`, en `_numero()`, al leer palabras float de `memoria.bin`.
- `src/isa.py`, en `ISA._constante()`, al parsear la constante de una
  instrucción `MOV`.

## Normalización retirada

`normalizar_numero()` ya no forma parte del código activo. Los floats de memoria
y las constantes `MOV` se conservan como floats. `formatear_numero()` solo
afecta cómo se muestran; no cambia los valores almacenados ni usados en los
cálculos.
