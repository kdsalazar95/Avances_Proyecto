# Guia de uso

## Requerimientos

| **Entorno** | **Gestor** |
|-------------|------------|
| Python      | UV         |


## Instrucciones

```bash
# Sincronizar requerimientos con uv

uv sync

# Activar entorno 

uv venv
```


__Importante:__ Revise si su sistema usa `python` o `python3` antes de ejecutar los scripts.


```bash
# Asegurarse de estar en el directorio Tarea_1

cd Tarea_1

# Correr algoritmo de Dijkstra

uv run python3 Dijkstra/main.py

# Correr algoritmo de RL

uv run python3 RL/train.py

# Generar resultados

uv run python3 RL/plot.py
```
