"""Definiciones compartidas de las direcciones de la malla CGRA."""

DIRECCIONES = ("north", "south", "east", "west")
OPUESTA = {"north": "south", "south": "north", "east": "west", "west": "east"}
DESPLAZAMIENTO = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
EN_ESPANOL = {"north": "norte", "south": "sur", "west": "oeste", "east": "este"}
