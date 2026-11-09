// Operaciones de prueba, distintas de las de operaciones/operaciones.c, con la
// misma estructura (inicializar_<operacion>, <operacion> y la tabla
// OPERACIONES). Sirven para probar que el flujo no depende de cuales son las
// operaciones: se compilan junto a una copia de memoria_cgra.h.

#include "memoria_cgra.h"

// ---------------------------------------------------------------------------
// Reduccion con multiplicacion: c[i] = a[i] * b[i] y result = producto de c.
// ---------------------------------------------------------------------------

#define TAMANO_PRODUCTO 12

static void inicializar_reduccion_producto(float a[TAMANO_PRODUCTO], float b[TAMANO_PRODUCTO],
                                           float c[TAMANO_PRODUCTO], float *result) {
    for (int i = 0; i < TAMANO_PRODUCTO; ++i) {
        a[i] = 1.0f + (float)(i % 3) * 0.125f;
        b[i] = 1.0f + (float)(i % 2) * 0.25f;
        c[i] = 0.0f;
    }
    *result = 1.0f;

    const RegionMemoria regiones[] = {
        {"a", a, 1, TAMANO_PRODUCTO},
        {"b", b, 1, TAMANO_PRODUCTO},
        {"c", c, 1, TAMANO_PRODUCTO},
        {"result", result, 1, 1},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void reduccion_producto(void) {
    float a[TAMANO_PRODUCTO];
    float b[TAMANO_PRODUCTO];
    float c[TAMANO_PRODUCTO];
    float result;

    inicializar_reduccion_producto(a, b, c, &result);

    for (int i = 0; i < TAMANO_PRODUCTO; ++i) {
        c[i] = a[i] * b[i];
    }

    for (int i = 0; i < TAMANO_PRODUCTO; ++i) {
        result *= c[i];
    }

    (void)result;
}

// ---------------------------------------------------------------------------
// Resta y division: salida[i] = (x[i] - y[i]) / 2.5 - 1.
// ---------------------------------------------------------------------------

#define TAMANO_RESTA 16

static void inicializar_resta_division(float x[TAMANO_RESTA], float y[TAMANO_RESTA], float salida[TAMANO_RESTA]) {
    for (int i = 0; i < TAMANO_RESTA; ++i) {
        x[i] = (float)(i + 3);
        y[i] = (float)(i % 5);
        salida[i] = 0.0f;
    }

    const RegionMemoria regiones[] = {
        {"x", x, 1, TAMANO_RESTA},
        {"y", y, 1, TAMANO_RESTA},
        {"salida", salida, 1, TAMANO_RESTA},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void resta_division(void) {
    float x[TAMANO_RESTA];
    float y[TAMANO_RESTA];
    float salida[TAMANO_RESTA];

    inicializar_resta_division(x, y, salida);

    for (int i = 0; i < TAMANO_RESTA; ++i) {
        salida[i] = (x[i] - y[i]) / 2.5f - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Producto mas constante y suma: c[i] = a[i] * b[i] + 1 (clang lo une en
// fmuladd) y total = suma de c.
// ---------------------------------------------------------------------------

#define TAMANO_MEZCLA 10

static void inicializar_mezcla(float a[TAMANO_MEZCLA], float b[TAMANO_MEZCLA], float c[TAMANO_MEZCLA],
                               float *total) {
    for (int i = 0; i < TAMANO_MEZCLA; ++i) {
        a[i] = (float)(i % 4 + 1);
        b[i] = (float)(i % 3 + 2);
        c[i] = 0.0f;
    }
    *total = 0.0f;

    const RegionMemoria regiones[] = {
        {"a", a, 1, TAMANO_MEZCLA},
        {"b", b, 1, TAMANO_MEZCLA},
        {"c", c, 1, TAMANO_MEZCLA},
        {"total", total, 1, 1},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void mezcla(void) {
    float a[TAMANO_MEZCLA];
    float b[TAMANO_MEZCLA];
    float c[TAMANO_MEZCLA];
    float total;

    inicializar_mezcla(a, b, c, &total);

    for (int i = 0; i < TAMANO_MEZCLA; ++i) {
        c[i] = a[i] * b[i] + 1.0f;
    }

    for (int i = 0; i < TAMANO_MEZCLA; ++i) {
        total += c[i];
    }

    (void)total;
}

// ---------------------------------------------------------------------------

typedef struct {
    const char *nombre;
    void (*ejecutar)(void);
} Operacion;

static const Operacion OPERACIONES[] = {
    {"reduccion_producto", reduccion_producto},
    {"resta_division", resta_division},
    {"mezcla", mezcla},
};

#define CANTIDAD_OPERACIONES (sizeof OPERACIONES / sizeof OPERACIONES[0])

int main(int argc, char **argv) {
    if (argc == 2) {
        for (size_t i = 0; i < CANTIDAD_OPERACIONES; ++i) {
            if (strcmp(argv[1], OPERACIONES[i].nombre) == 0) {
                OPERACIONES[i].ejecutar();
                return 0;
            }
        }
    }

    fprintf(stderr, "Uso: %s <operacion>\nOperaciones:", argv[0]);
    for (size_t i = 0; i < CANTIDAD_OPERACIONES; ++i) {
        fprintf(stderr, " %s", OPERACIONES[i].nombre);
    }
    fprintf(stderr, "\n");
    return 1;
}
