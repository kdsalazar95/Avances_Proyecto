"""
Configuracion de pytest.

'test/' es hermana de 'Dijkstra/', 'RL/' y 'comun/' (no las contiene),
asi que hay que agregar las TRES carpetas al path de Python para que
los tests puedan hacer 'from grafo import Grafo', 'from agente import ...',
etc. sin necesidad de instalar nada como paquete.

'comun/' contiene grafo.py, compartido por Dijkstra/ y RL/ (una sola
fuente de verdad, en vez de dos copias independientes).
"""

import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _carpeta in ("Dijkstra", "RL", "comun"):
    sys.path.insert(0, os.path.join(_RAIZ, _carpeta))