# Instrucciones del agente RL: proponer la configuración de la CGRA

## Propósito

Este archivo lo lee el LLM que hace de agente del RL (el prompt que arma `rl/agente/politica.py` le pide seguirlo). Su acción es la configuración de la CGRA. Según el diseño del proyecto (PDF "Dijkstra + Greedy _ Agente RL", página 1), las acciones son:

- **Malla de PEs**: el mapa de distribución (qué tipo de PE va en cada posición).
- **Tipos de PEs**: memoria, escalar, vectorial, cómputo, genérico (los del `cgra.json` real).
- **Tamaño de red**: filas × columnas, dentro del rango de la corrida.

El agente decide *qué CGRA construir*. *Cómo ejecutar el cómputo en esa CGRA* (mapeo, calendarización, `PE*.txt`) lo decide el evaluador con los Skills 01–09, que ya existían.

## Entradas

- El contexto del problema (`contexto.json`): operación/instancia, regiones de memoria, tipos e instrucciones reales, restricciones.
- La entrada de la política del enunciado, $\pi(C, E) = A$ (viene en el prompt):
  - el costo $C = K_A C_A + K_L C_L + K_R C_R$, con los $K$ fijados desde fuera;
  - el estado $E$: los cambios que produjo tu última acción respecto al mejor candidato del que partió, es decir, la malla antes y después y $C_A$, $C_L$, $C_R$ y $C$ antes, después y su cambio, más la recompensa $r$ con la que se la premió o penalizó;
  - el mejor candidato actual y sus costos.
- Parámetros de corrida: proxies de área $w$, prioridades $K$, normalización, penalización.

## Acción propuesta

En el directorio que indica el ciclo (`corridas/<id>/candidatas/<id_candidata>/`):

1. `cgra.json` con el formato de `FormatoCGRA`: `filas` y `columnas` enteros, `palabras_memoria`, `ancho_vector`, `tipos_pe` (solo los usados, con sus listas exactas del esquema) y una `malla` explícita (una cadena por fila, una letra por PE).
2. `candidata.json`: `id_candidata` y una justificación breve basada en el costo $C$, el estado $E$ y el mejor candidato.

No se entregan `PE*.txt` ni manifest de asignación: la validación previa (`rl/entorno/validacion_accion.py`) rechaza una acción que traiga `pe_instructions/`.

## Procedimiento

1. Mira el estado $E$ (qué cambiaste en la malla y cómo cambiaron $C_A$, $C_L$, $C_R$ y $C$) y el mejor candidato actual. Regla del enunciado: si una acción da un costo menor que el del mejor candidato, pasa a ser el mejor candidato; si da un costo mayor, se descarta.
2. Decide hacia dónde moverte para minimizar $C$ según los $K$: por ejemplo, menos PEs si pesa el área; más filas de cómputo o PEs vectoriales si pesa la latencia.
3. Decide solo con la información del prompt: no leas el historial, los reportes de evaluación ni otras candidatas.
4. Respeta las reglas de la corrida: las columnas oeste y este son todas de memoria (`M`) y el interior es solo cómputo (por eso hacen falta al menos 3 columnas); al menos un PE de cómputo; tamaño dentro del rango.
5. **Tú eliges el tipo de cada PE del interior** (`E`, `V` o `C`), sin seguir una rotación fija: puedes mezclar tipos en una misma fila o columna. Ten en cuenta el costo (un `C` suma el área de un `E` y un `V`) y lo que necesita la operación (cómputo vectorial en `V`/`C`, el resto escalar en `E`/`C`, y rutas compatibles hasta la memoria, ver `cgra/agent_skills/mesh-compatibility.md`).
6. Escribe los dos archivos y termina. No ejecutes el evaluador ni el simulador.

## Reglas

- No inventes tipos, instrucciones ni capacidades fuera del esquema real.
- No modifiques fuente, entradas, historial, reportes ni candidatas ya guardadas.
- No declares una configuración correcta ni le asignes costos: eso lo mide el evaluador.
