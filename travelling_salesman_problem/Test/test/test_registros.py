"""
Tests de registro.py y registro_busqueda.py

Verifican que las clases de registro acumulen filas en memoria
correctamente y que guarden un CSV valido y legible en disco.
Usan el fixture 'tmp_path' de pytest, que crea una carpeta temporal
descartable para no ensuciar el proyecto con archivos de prueba.
"""

import csv

from registro import RegistroEntrenamiento
from reg_busqueda import RegistroBusqueda


def test_registro_entrenamiento_acumula_filas():
    registro = RegistroEntrenamiento()
    registro.registrar(episodio=0, recompensa_total=-10, costo_total=10, pasos=3, epsilon=0.5)
    registro.registrar(episodio=1, recompensa_total=-8, costo_total=8, pasos=3, epsilon=0.4)
    assert len(registro.filas) == 2
    assert registro.filas[0]["episodio"] == 0


def test_registro_entrenamiento_guarda_csv_legible(tmp_path):
    registro = RegistroEntrenamiento()
    registro.registrar(episodio=0, recompensa_total=-10, costo_total=10, pasos=3, epsilon=0.5)
    ruta = tmp_path / "historial.csv"

    registro.guardar_csv(ruta)

    assert ruta.exists()
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 1
    assert filas[0]["episodio"] == "0"
    assert filas[0]["costo_total"] == "10"


def test_registro_entrenamiento_resumen_filtra_cada_n_episodios():
    registro = RegistroEntrenamiento()
    for i in range(10):
        registro.registrar(episodio=i, recompensa_total=0, costo_total=0, pasos=1, epsilon=0.1)
    resumen = registro.resumen(cada_n=5)
    assert [f["episodio"] for f in resumen] == [0, 5]


def test_registro_busqueda_acumula_y_guarda_csv(tmp_path):
    registro = RegistroBusqueda()
    registro.registrar(iteracion=0, estado=(0, 1), costo_acumulado=0, tamano_cola=2)
    registro.registrar(iteracion=1, estado=(1, 3), costo_acumulado=5, tamano_cola=4)

    ruta = tmp_path / "busqueda.csv"
    registro.guardar_csv(ruta)

    assert ruta.exists()
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 2
    assert filas[1]["costo_acumulado"] == "5"


def test_guardar_csv_sin_filas_no_falla(tmp_path):
    registro = RegistroEntrenamiento()
    ruta = tmp_path / "vacio.csv"
    registro.guardar_csv(ruta)  # no deberia lanzar excepcion
    assert not ruta.exists()  # no se crea archivo si no hay nada que guardar
