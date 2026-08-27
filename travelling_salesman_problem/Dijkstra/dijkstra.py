"""
Modulo del ALGORITMO.

Implementacion generica de Dijkstra sobre estados abstractos.
No sabe nada de ciudades, matrices de distancia, ni TSP: solo necesita
un objeto "problema" que le diga:
    - problema.estado_inicial()      -> el estado de partida
    - problema.vecinos(estado)       -> [(estado_siguiente, costo), ...]

Esto significa que este mismo modulo podria reutilizarse para resolver
cualquier otro problema de camino mas corto, no solo el TSP.
"""

import heapq


def dijkstra(problema, registro=None):
    """
    Ejecuta Dijkstra sobre el grafo de estados definido por 'problema'.

    registro: instancia opcional de RegistroBusqueda (ver registro_busqueda.py).
              Si se pasa, se guarda una fila por cada estado expandido
              (iteracion, estado, costo acumulado, tamano de la cola).

    Devuelve dos diccionarios:
      - costos: costos[estado] = menor costo conocido para llegar a ese estado
      - padres: padres[estado] = estado anterior en el mejor camino encontrado
                (para poder reconstruir la ruta despues)
    """
    inicio = problema.estado_inicial()

    costos = {inicio: 0}
    padres = {}

    # Cola de prioridad: (costo_acumulado, estado)
    # Usamos un contador auxiliar para evitar comparar estados directamente
    # si dos costos son iguales (heapq compararia el segundo elemento).
    contador = 0
    heap = [(0, contador, inicio)]

    visitados = set()
    iteracion = 0

    while heap:
        costo, _, estado = heapq.heappop(heap)

        if estado in visitados:
            continue
        visitados.add(estado)

        if registro is not None:
            registro.registrar(
                iteracion=iteracion,
                estado=estado,
                costo_acumulado=costo,
                tamano_cola=len(heap),
            )
            iteracion += 1

        for vecino, peso_arista in problema.vecinos(estado):
            nuevo_costo = costo + peso_arista
            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = estado
                contador += 1
                heapq.heappush(heap, (nuevo_costo, contador, vecino))

    return costos, padres


def reconstruir_camino(padres, estado_final, estado_inicial):
    """
    Sigue los punteros 'padres' desde 'estado_final' hasta 'estado_inicial'
    para reconstruir la secuencia completa de estados del camino optimo.
    """
    camino = [estado_final]
    actual = estado_final
    while actual != estado_inicial:
        actual = padres[actual]
        camino.append(actual)
    camino.reverse()
    return camino