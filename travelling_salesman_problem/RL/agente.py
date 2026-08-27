"""
Modulo del AGENTE.

El agente es quien "decide" (politica epsilon-greedy) y quien "aprende"
(regla de actualizacion de Q-learning). No sabe nada de TSP ni de
ciudades: solo recibe estados y acciones genericos, y una recompensa
tras cada paso. Se le puede acoplar a CUALQUIER entorno que entregue
estados/acciones en el formato esperado (no solo entorno_tsp.py).

Esta separacion permite:
    - ajustar los hiperparametros del agente sin tocar el bucle de
      entrenamiento (ver q_learning.py) ni el entorno.
    - crear otros tipos de agente en el futuro (ej. SARSA, Double
      Q-learning) que se puedan usar de forma intercambiable, ya que
      todos expondrian los mismos metodos: elegir_accion, actualizar,
      decaer_epsilon.
"""

import random
from collections import defaultdict


class AgenteQLearning:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon_inicial=0.3,
                 epsilon_minimo=0.01, decaimiento_epsilon=0.999, semilla=None):
        """
        alpha: tasa de aprendizaje (cuanto se ajusta Q en cada actualizacion).
        gamma: factor de descuento (1.0 = sin descuento; apropiado para
               episodios finitos y cortos como este).
        epsilon_inicial: probabilidad inicial de explorar (accion aleatoria).
        epsilon_minimo: piso al que puede bajar epsilon (nunca deja de explorar del todo).
        decaimiento_epsilon: factor que multiplica a epsilon tras cada episodio.
        semilla: semilla para el generador aleatorio (reproducibilidad).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_inicial
        self.epsilon_minimo = epsilon_minimo
        self.decaimiento_epsilon = decaimiento_epsilon

        self.Q = defaultdict(float)
        self._rng = random.Random(semilla)

    def elegir_accion(self, estado, acciones_validas):
        """
        Politica epsilon-greedy: con probabilidad epsilon elige una accion
        aleatoria (explorar), si no elige la de mayor valor Q conocido
        (explotar lo aprendido).
        """
        if self._rng.random() < self.epsilon:
            return self._rng.choice(acciones_validas)
        return self.mejor_accion(estado, acciones_validas)

    def mejor_accion(self, estado, acciones_validas):
        """Accion de mayor valor Q conocido para 'estado' (sin exploracion)."""
        return max(acciones_validas, key=lambda a: self.Q[(estado, a)])

    def actualizar(self, estado, accion, recompensa, nuevo_estado,
                   acciones_siguientes, terminado):
        """
        Aplica la regla de actualizacion de Q-learning:
            Q(s,a) <- Q(s,a) + alpha * (recompensa + gamma * max_a' Q(s',a') - Q(s,a))
        """
        if terminado:
            mejor_q_siguiente = 0.0
        else:
            mejor_q_siguiente = max(self.Q[(nuevo_estado, a)] for a in acciones_siguientes)

        objetivo = recompensa + self.gamma * mejor_q_siguiente
        self.Q[(estado, accion)] += self.alpha * (objetivo - self.Q[(estado, accion)])

    def decaer_epsilon(self):
        """Reduce epsilon tras cada episodio, sin bajar del piso minimo."""
        self.epsilon = max(self.epsilon_minimo, self.epsilon * self.decaimiento_epsilon)

    @classmethod
    def desde_config(cls, config):
        """Crea un agente a partir de un diccionario de hiperparametros (ej. de un JSON)."""
        return cls(
            alpha=config["alpha"],
            gamma=config["gamma"],
            epsilon_inicial=config["epsilon_inicial"],
            epsilon_minimo=config.get("epsilon_minimo", 0.01),
            decaimiento_epsilon=config["decaimiento_epsilon"],
            semilla=config.get("semilla"),
        )