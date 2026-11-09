// Referencia de las pruebas: corre una operacion de operaciones.c y guarda la
// memoria FINAL (despues del calculo) en memoria_final.bin, con el mismo
// formato que memoria.bin. Asi las pruebas comparan contra lo que calcula el
// propio C en float32, sin escribir cada operacion otra vez a mano.
//
// No cambia operaciones.c: lo incluye, y se compila con -finstrument-functions
// para que __cyg_profile_func_exit se llame justo antes de que <operacion>()
// termine, con sus arreglos todavia vivos:
//
//   clang-18 -finstrument-functions -I <carpeta de operaciones.c> \
//            -DFUENTE='"operaciones.c"' memoria_final.c -o memoria_final.out
//   ./memoria_final.out <operacion>     # escribe memoria.bin y memoria_final.bin

#include "memoria_cgra.h"

#define ARCHIVO_FINAL "memoria_final.bin"
#define MAXIMO_REGIONES 64

static RegionMemoria regiones_finales[MAXIMO_REGIONES];
static uint32_t cantidad_finales;
static void (*operacion_actual)(void);

// inicializar_<operacion>() guarda la memoria inicial como siempre; ademas se
// recuerdan las regiones (punteros a los arreglos de <operacion>()).
static void guardar_y_recordar(const char *ruta, const RegionMemoria *regiones, uint32_t cantidad) {
    guardar_memoria(ruta, regiones, cantidad);
    if (cantidad > MAXIMO_REGIONES) {
        fprintf(stderr, "mas de %d regiones\n", MAXIMO_REGIONES);
        exit(1);
    }
    memcpy(regiones_finales, regiones, cantidad * sizeof regiones[0]);
    cantidad_finales = cantidad;
}

#define guardar_memoria guardar_y_recordar
#define main main_operaciones
#include FUENTE
#undef main
#undef guardar_memoria

__attribute__((no_instrument_function)) void __cyg_profile_func_enter(void *funcion, void *llamada) {
    (void)funcion;
    (void)llamada;
}

__attribute__((no_instrument_function)) void __cyg_profile_func_exit(void *funcion, void *llamada) {
    (void)llamada;
    if (operacion_actual && funcion == (void *)operacion_actual && cantidad_finales) {
        operacion_actual = NULL;
        guardar_memoria(ARCHIVO_FINAL, regiones_finales, cantidad_finales);
    }
}

int main(int argc, char **argv) {
    for (size_t i = 0; argc == 2 && i < CANTIDAD_OPERACIONES; ++i) {
        if (strcmp(argv[1], OPERACIONES[i].nombre) == 0) {
            operacion_actual = OPERACIONES[i].ejecutar;
            OPERACIONES[i].ejecutar();
            return 0;
        }
    }
    fprintf(stderr, "Uso: %s <operacion>\n", argv[0]);
    return 1;
}
