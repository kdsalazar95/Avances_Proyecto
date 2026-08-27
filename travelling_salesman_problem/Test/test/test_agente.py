"""
Tests de agente.py

Verifican la logica del agente de Q-learning de forma aislada: la
politica epsilon-greedy, la regla de actualizacion de Q, y el
decaimiento de epsilon. No usa el entorno TSP, solo estados y
acciones genericos (numeros).
"""

from agente import AgenteQLearning


def test_con_epsilon_cero_siempre_elige_la_mejor_accion_conocida():
    agente = AgenteQLearning(epsilon_inicial=0.0, semilla=1)
    agente.Q[("s", "accion_buena")] = 10.0
    agente.Q[("s", "accion_mala")] = 1.0

    accion = agente.elegir_accion("s", ["accion_buena", "accion_mala"])
    assert accion == "accion_buena"


def test_con_epsilon_uno_siempre_explora_al_azar():
    agente = AgenteQLearning(epsilon_inicial=1.0, semilla=1)
    agente.Q[("s", "a1")] = 100.0  # aunque una accion sea mucho mejor...
    acciones_elegidas = {
        agente.elegir_accion("s", ["a1", "a2", "a3"]) for _ in range(30)
    }
    # con epsilon=1 y suficientes intentos, deberia explorar mas de una opcion
    assert len(acciones_elegidas) > 1


def test_actualizar_acerca_q_hacia_el_objetivo():
    agente = AgenteQLearning(alpha=0.5, gamma=1.0)
    agente.Q[("s", "a")] = 0.0

    agente.actualizar(
        estado="s", accion="a", recompensa=10,
        nuevo_estado="s2", acciones_siguientes=[], terminado=True,
    )
    # objetivo = 10 + 0 (terminal); Q se mueve la mitad del camino (alpha=0.5)
    assert agente.Q[("s", "a")] == 5.0


def test_actualizar_usa_el_mejor_q_del_siguiente_estado_si_no_es_terminal():
    agente = AgenteQLearning(alpha=1.0, gamma=1.0)  # alpha=1 => copia el objetivo exacto
    agente.Q[("s2", "a1")] = 3.0
    agente.Q[("s2", "a2")] = 7.0  # esta es la mejor

    agente.actualizar(
        estado="s", accion="a", recompensa=1,
        nuevo_estado="s2", acciones_siguientes=["a1", "a2"], terminado=False,
    )
    # objetivo = 1 + 7 (la mejor accion siguiente) = 8
    assert agente.Q[("s", "a")] == 8.0


def test_decaer_epsilon_reduce_pero_respeta_el_minimo():
    agente = AgenteQLearning(epsilon_inicial=0.1, epsilon_minimo=0.05,
                              decaimiento_epsilon=0.5)
    agente.decaer_epsilon()
    assert agente.epsilon == 0.05  # 0.1 * 0.5 = 0.05, justo el minimo
    agente.decaer_epsilon()
    assert agente.epsilon == 0.05  # no baja mas del minimo


def test_desde_config_crea_agente_con_los_valores_del_diccionario():
    config = {
        "alpha": 0.2, "gamma": 0.9, "epsilon_inicial": 0.5,
        "epsilon_minimo": 0.02, "decaimiento_epsilon": 0.99, "semilla": 7,
    }
    agente = AgenteQLearning.desde_config(config)
    assert agente.alpha == 0.2
    assert agente.gamma == 0.9
    assert agente.epsilon == 0.5
    assert agente.epsilon_minimo == 0.02
