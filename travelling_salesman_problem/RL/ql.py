"""
Modulo del ENTRENAMIENTO.

Bucle generico que hace interactuar a un AGENTE (agente.py) con un
ENTORNO (ej. entorno_tsp.py), episodio tras episodio. No sabe nada de
TSP ni de ciudades: solo conoce la interfaz del agente
(elegir_accion, actualizar, decaer_epsilon) y la del entorno
(reset, paso, acciones_validas).

Esto significa que este mismo bucle serviria para entrenar al agente
en cualquier otro entorno, sin cambiar una linea.
"""


def entrenar(entorno, agente, episodios, registro=None):
    """
    Corre 'episodios' episodios completos de interaccion agente-entorno,
    dejando que el agente aprenda con cada paso.

    registro: instancia opcional de RegistroEntrenamiento (ver registro.py).
              Si se pasa, se guardan las metricas de cada episodio
              (recompensa total, costo, pasos, epsilon usado).
    """
    for episodio in range(episodios):
        estado = entorno.reset()
        terminado = False

        recompensa_total = 0.0
        pasos = 0
        epsilon_del_episodio = agente.epsilon

        while not terminado:
            acciones = entorno.acciones_validas(estado)
            accion = agente.elegir_accion(estado, acciones)

            nuevo_estado, recompensa, terminado = entorno.paso(accion)

            acciones_siguientes = (
                [] if terminado else entorno.acciones_validas(nuevo_estado)
            )
            agente.actualizar(estado, accion, recompensa, nuevo_estado,
                               acciones_siguientes, terminado)

            recompensa_total += recompensa
            pasos += 1
            estado = nuevo_estado

        if registro is not None:
            registro.registrar(
                episodio=episodio,
                recompensa_total=recompensa_total,
                costo_total=-recompensa_total,
                pasos=pasos,
                epsilon=epsilon_del_episodio,
            )

        agente.decaer_epsilon()

    return agente


def extraer_politica_greedy(agente, entorno):
    """
    Usa lo aprendido por el agente para recorrer el entorno tomando siempre
    la mejor accion conocida (sin exploracion). Devuelve (ruta, costo_total).
    """
    estado = entorno.reset()
    ruta = [estado[0]]
    costo_total = 0.0
    terminado = False

    while not terminado:
        acciones = entorno.acciones_validas(estado)
        accion = agente.mejor_accion(estado, acciones)
        nuevo_estado, recompensa, terminado = entorno.paso(accion)
        costo_total += -recompensa
        ruta.append(accion)
        estado = nuevo_estado

    return ruta, costo_total