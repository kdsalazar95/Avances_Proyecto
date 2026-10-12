/*
 * Memoria de la CGRA en binario.
 *
 * Cada operacion de operaciones.c (reduccion, matmul, convolucion) crea e
 * inicializa sus arreglos en inicializar_<operacion>() y al final los guarda
 * con guardar_memoria(). El archivo resultante, memoria.bin, es la memoria
 * inicial que usan los skills y que carga run_cgra.py.
 *
 * Formato (little-endian, igual que la memoria del procesador que lo
 * genera):
 *
 *   char     firma[8]        "CGRAMEM\0"
 *   uint32   cantidad        número de regiones
 *   uint32   palabras        tamaño total de la memoria, en floats
 *   cantidad x {             tabla de símbolos
 *     char   nombre[16]      banco que usan los LD/ST de los PE*.txt
 *     uint32 direccion       primera palabra de la región
 *     uint32 filas
 *     uint32 columnas
 *   }
 *   float32  datos[palabras] memoria plana; las matrices, fila por fila
 */

#ifndef MEMORIA_CGRA_H
#define MEMORIA_CGRA_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARCHIVO_MEMORIA "memoria.bin"
#define LARGO_NOMBRE_REGION 16

typedef struct {
    const char *nombre;
    const float *datos;
    uint32_t filas;
    uint32_t columnas;
} RegionMemoria;

static void escribir(FILE *archivo, const void *datos, size_t tamano, size_t cantidad) {
    if (fwrite(datos, tamano, cantidad, archivo) != cantidad) {
        perror(ARCHIVO_MEMORIA);
        exit(1);
    }
}

static void guardar_memoria(const char *ruta, const RegionMemoria *regiones, uint32_t cantidad) {
    FILE *archivo = fopen(ruta, "wb");
    if (!archivo) {
        perror(ruta);
        exit(1);
    }

    uint32_t palabras = 0;
    for (uint32_t r = 0; r < cantidad; ++r) {
        palabras += regiones[r].filas * regiones[r].columnas;
    }
    escribir(archivo, "CGRAMEM", 1, 8);
    escribir(archivo, &cantidad, sizeof cantidad, 1);
    escribir(archivo, &palabras, sizeof palabras, 1);

    uint32_t direccion = 0;
    for (uint32_t r = 0; r < cantidad; ++r) {
        char nombre[LARGO_NOMBRE_REGION] = {0};
        strncpy(nombre, regiones[r].nombre, LARGO_NOMBRE_REGION - 1);
        escribir(archivo, nombre, 1, LARGO_NOMBRE_REGION);
        escribir(archivo, &direccion, sizeof direccion, 1);
        escribir(archivo, &regiones[r].filas, sizeof regiones[r].filas, 1);
        escribir(archivo, &regiones[r].columnas, sizeof regiones[r].columnas, 1);
        direccion += regiones[r].filas * regiones[r].columnas;
    }
    for (uint32_t r = 0; r < cantidad; ++r) {
        escribir(archivo, regiones[r].datos, sizeof(float), regiones[r].filas * regiones[r].columnas);
    }

    fclose(archivo);
    printf("Memoria inicial guardada en %s (%u regiones, %u palabras)\n", ruta, cantidad, palabras);
}

#endif
