"""Etapa independiente: compilar C a LLVM IR y generar memoria inicial."""
import argparse
from dijkstra import operaciones as operaciones_del_c
from dijkstra.compilar import preparar, ErrorCompilacion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operacion", help="Nombre de operación o 'todas'")
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()
    operaciones = operaciones_del_c() if args.operacion == "todas" else [args.operacion]
    try:
        hechas = preparar(operaciones, forzar=args.forzar)
    except ErrorCompilacion as error:
        raise SystemExit(str(error))
    print("Compiladas: " + ", ".join(hechas) if hechas else "IR y memoria inicial ya están al día.")


if __name__ == "__main__":
    main()
