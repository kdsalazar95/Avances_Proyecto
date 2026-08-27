"""
Configuracion de pytest.

Agrega la carpeta raiz del proyecto (donde estan grafo.py, dijkstra.py,
etc.) al path de Python, para que los tests puedan hacer
'from grafo import Grafo' sin necesidad de instalar nada como paquete.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
