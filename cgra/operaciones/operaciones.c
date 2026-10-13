// Operaciones que se compilan para la CGRA: reduccion, matmul y convolucion.
//
// Cada operacion tiene dos funciones:
//
//   inicializar_<operacion>()  crea e inicializa sus arreglos y los guarda en
//                              memoria.bin, la memoria inicial de la CGRA. Solo
//                              prepara los datos: los skills no la calendarizan.
//   <operacion>()              el trabajo de la CGRA: llama a su
//                              inicializar_<operacion>() y despues corre los
//                              ciclos que los skills calendarizan.
//
// main() solo elige la operacion por su nombre:
//
//   clang-18 operaciones.c -o operaciones.out && ./operaciones.out matmul
//
// escribe memoria.bin (la de matmul) en el directorio actual.

#include "memoria_cgra.h"

// ---------------------------------------------------------------------------
// Reduccion: c[i] = a[i] + b[i] y result = suma de c.
// ---------------------------------------------------------------------------

#define TAMANO_VECTOR 100

static void inicializar_reduccion(float a[TAMANO_VECTOR], float b[TAMANO_VECTOR], float c[TAMANO_VECTOR],
                                  float *result) {
    for (int i = 0; i < TAMANO_VECTOR; ++i) {
        a[i] = (float)(i % 4 + 1);
        b[i] = (float)(i % 4 + 1);
        c[i] = 0.0f;
    }
    *result = 0.0f;

    const RegionMemoria regiones[] = {
        {"a", a, 1, TAMANO_VECTOR},
        {"b", b, 1, TAMANO_VECTOR},
        {"c", c, 1, TAMANO_VECTOR},
        {"result", result, 1, 1},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void reduccion(void) {
    float a[TAMANO_VECTOR];
    float b[TAMANO_VECTOR];
    float c[TAMANO_VECTOR];
    float result;

    inicializar_reduccion(a, b, c, &result);

    for (int i = 0; i < TAMANO_VECTOR; ++i) {
        c[i] = a[i] + b[i];
    }

    for (int i = 0; i < TAMANO_VECTOR; ++i) {
        result += c[i];
    }

    (void)result;
}

// ---------------------------------------------------------------------------
// Matmul: C = A x B. La matriz c se guarda en el banco "result".
// ---------------------------------------------------------------------------

#define TAMANO_MATRIZ 9 // Ajustar tamano para que coincida con el tamano de la matriz

static void inicializar_matmul(float a[TAMANO_MATRIZ][TAMANO_MATRIZ], float b[TAMANO_MATRIZ][TAMANO_MATRIZ],
                               float c[TAMANO_MATRIZ][TAMANO_MATRIZ]) {
    for (int i = 0; i < TAMANO_MATRIZ; ++i) {
        for (int j = 0; j < TAMANO_MATRIZ; ++j) {
            a[i][j] = (float)((i + j) % 5 + 1);
            b[i][j] = (float)((i + j) % 3 + 1);
            c[i][j] = 0.0f;
        }
    }

    const RegionMemoria regiones[] = {
        {"a", &a[0][0], TAMANO_MATRIZ, TAMANO_MATRIZ},
        {"b", &b[0][0], TAMANO_MATRIZ, TAMANO_MATRIZ},
        {"result", &c[0][0], TAMANO_MATRIZ, TAMANO_MATRIZ},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void matmul(void) {
    float a[TAMANO_MATRIZ][TAMANO_MATRIZ];
    float b[TAMANO_MATRIZ][TAMANO_MATRIZ];
    float c[TAMANO_MATRIZ][TAMANO_MATRIZ];

    inicializar_matmul(a, b, c);

    for (int i = 0; i < TAMANO_MATRIZ; ++i) {
        for (int j = 0; j < TAMANO_MATRIZ; ++j) {
            float acc = 0.f;
            for (int k = 0; k < TAMANO_MATRIZ; ++k) {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Convolucion: result[i][j] = suma de imagen[i+ki][j+kj] * kernel[ki][kj].
// La salida se guarda en el banco "result".
// ---------------------------------------------------------------------------

#define TAMANO_IMAGEN 12
#define TAMANO_KERNEL 4

#define TAMANO_SALIDA (TAMANO_IMAGEN - TAMANO_KERNEL + 1)

static void inicializar_convolucion(float imagen[TAMANO_IMAGEN][TAMANO_IMAGEN],
                                    float kernel[TAMANO_KERNEL][TAMANO_KERNEL],
                                    float resultado[TAMANO_SALIDA][TAMANO_SALIDA]) {
    for (int i = 0; i < TAMANO_IMAGEN; ++i) {
        for (int j = 0; j < TAMANO_IMAGEN; ++j) {
            imagen[i][j] = (float)((i + j) % 7 + 1);
        }
    }

    for (int i = 0; i < TAMANO_KERNEL; ++i) {
        for (int j = 0; j < TAMANO_KERNEL; ++j) {
            kernel[i][j] = (float)((i + j) % 3 + 1);
        }
    }

    for (int i = 0; i < TAMANO_SALIDA; ++i) {
        for (int j = 0; j < TAMANO_SALIDA; ++j) {
            resultado[i][j] = 0.0f;
        }
    }

    const RegionMemoria regiones[] = {
        {"imagen", &imagen[0][0], TAMANO_IMAGEN, TAMANO_IMAGEN},
        {"kernel", &kernel[0][0], TAMANO_KERNEL, TAMANO_KERNEL},
        {"result", &resultado[0][0], TAMANO_SALIDA, TAMANO_SALIDA},
    };
    guardar_memoria(ARCHIVO_MEMORIA, regiones, sizeof regiones / sizeof regiones[0]);
}

static void convolucion(void) {
    float imagen[TAMANO_IMAGEN][TAMANO_IMAGEN];
    float kernel[TAMANO_KERNEL][TAMANO_KERNEL];
    float resultado[TAMANO_SALIDA][TAMANO_SALIDA];

    inicializar_convolucion(imagen, kernel, resultado);

    for (int i = 0; i < TAMANO_SALIDA; ++i) {
        for (int j = 0; j < TAMANO_SALIDA; ++j) {
            float acumulado = 0.0f;
            for (int ki = 0; ki < TAMANO_KERNEL; ++ki) {
                for (int kj = 0; kj < TAMANO_KERNEL; ++kj) {
                    acumulado += imagen[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            resultado[i][j] = acumulado;
        }
    }
}

// ---------------------------------------------------------------------------
// Seleccion de la operacion. Para agregar una operacion nueva: escribir sus
// dos funciones arriba y agregarla a esta tabla.
// ---------------------------------------------------------------------------

typedef struct {
    const char *nombre;
    void (*ejecutar)(void);
} Operacion;

static const Operacion OPERACIONES[] = {
    {"reduccion", reduccion},
    {"matmul", matmul},
    {"convolucion", convolucion},
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
