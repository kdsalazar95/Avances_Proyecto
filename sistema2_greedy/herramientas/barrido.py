"""Barrido de epsilon (y de paciencia) del Sistema 2: una corrida por epsilon con la misma configuración y semilla.

    python3 herramientas/barrido.py --epsilons 0 10 20 80 90 100            # desde sistema2_greedy/
    python3 herramientas/barrido.py --epsilons 0 10 20 80 90 100 --id s2-comparacion -- --procesos 3

Cada corrida es ``agente_greedy.py --epsilon E --paciencia 10 --max-episodios 30 --salida
resultados/<id>-eps<NNN>`` (``eps010`` = epsilon 0.10), con ``config/agente.json`` para todo lo demás. Lo que va
después de ``--`` se pasa a ``agente_greedy.py``.

Paciencia: cada corrida usa la mayor (``--paciencia``, 10) con un tope de episodios que no la limite
(``--max-episodios``, 30). Los resultados de paciencia 3 y 6 salen de cortar esas mismas corridas donde se habrían
detenido (``convergencia.corte``): 6 corridas dan las 18 combinaciones de epsilon y paciencia.

La referencia 32x32 se simula una sola vez (en la primera corrida) y cada corrida empieza con las
simulaciones de las anteriores (``evaluacion_malla.json`` y ``evaluaciones_episodios.json``): una malla ya
simulada no se vuelve a simular. Esto no cambia los datos de convergencia, porque cada evaluación cuenta lo
que tardó cuando se simuló (``herramientas/convergencia.py``). Una corrida con ``convergencia.json`` ya está
hecha y no se repite; una interrumpida se continúa.

Al final escribe la comparación en ``resultados/<id>-comparacion/`` (``comparacion_epsilon.csv``,
``curvas_epsilon.csv``), en el mismo formato que el Sistema 1.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ_DG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_DG))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARCHIVO_EVALUACION = "evaluacion_malla.json"
ARCHIVO_CACHE = "evaluaciones_episodios.json"


def leer_epsilons(valores):
    """``["0", "10", "0.2"]`` -> ``[0.0, 0.1, 0.2]``: los valores mayores que 1 son porcentajes."""
    epsilons = []
    for valor in valores:
        e = float(valor)
        e = e / 100 if e > 1 else e
        if not 0 <= e <= 1:
            raise ValueError(f"epsilon fuera de rango: {valor}")
        epsilons.append(round(e, 6))
    return epsilons


def nombre(id_base, epsilon):
    return f"{id_base}-eps{round(epsilon * 100):03d}"


def heredar(origen, destino):
    """Copia a ``destino`` la simulación de la referencia de ``origen`` (si no tiene una) y le agrega las
    simulaciones de mallas de ``origen`` que no tenga (misma huella: mismo programa, costos y normalizadores)."""
    destino.mkdir(parents=True, exist_ok=True)
    if (origen / ARCHIVO_EVALUACION).exists() and not (destino / ARCHIVO_EVALUACION).exists():
        shutil.copy2(origen / ARCHIVO_EVALUACION, destino / ARCHIVO_EVALUACION)
    if not (origen / ARCHIVO_CACHE).exists():
        return
    previa = json.loads((origen / ARCHIVO_CACHE).read_text(encoding="utf-8"))
    if (destino / ARCHIVO_CACHE).exists():
        propia = json.loads((destino / ARCHIVO_CACHE).read_text(encoding="utf-8"))
        if propia.get("huella") != previa.get("huella"):
            return
    else:
        propia = {"episodios": [], "huella": previa.get("huella"), "marca_tiempo": previa.get("marca_tiempo")}
    claves = {json.dumps(e["configuracion"], sort_keys=True) for e in propia["episodios"]}
    propia["episodios"] += [e for e in previa["episodios"]
                            if json.dumps(e["configuracion"], sort_keys=True) not in claves]
    (destino / ARCHIVO_CACHE).write_text(json.dumps(propia, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def barrido(epsilons, id_base="s2-comparacion", resultados=RAIZ_DG / "resultados", extra=(), correr=None,
            paciencia=10, max_episodios=30, paciencias=None):
    """Corre (o continúa) una corrida por epsilon y escribe la comparación. Devuelve ``(carpetas, resumenes)``.
    ``paciencias``: las que se calculan cortando cada corrida (por defecto ``convergencia.PACIENCIAS``).

    ``correr(argv)``: el ``main`` de ``agente_greedy.py`` (se puede cambiar en las pruebas)."""
    import convergencia
    if correr is None:
        import agente_greedy
        correr = agente_greedy.main
    resultados = Path(resultados)
    carpetas, anterior = [], None
    for epsilon in epsilons:
        carpeta = resultados / nombre(id_base, epsilon)
        if not (carpeta / "convergencia.json").exists():
            if anterior is not None:
                heredar(anterior, carpeta)
            codigo = correr(["--salida", str(carpeta), "--epsilon", str(epsilon), "--paciencia", str(paciencia),
                             "--max-episodios", str(max_episodios), "--continuar", *extra])
            if codigo:
                raise RuntimeError(f"la corrida de epsilon {epsilon} terminó con código {codigo}")
        carpetas.append(carpeta)
        anterior = carpeta
    resumenes = convergencia.comparar(carpetas, resultados / f"{id_base}-comparacion",
                                      paciencias or convergencia.PACIENCIAS)
    return carpetas, resumenes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", required=True, help="0 a 1, o porcentajes: 0 10 20 80 90 100")
    parser.add_argument("--id", default="s2-comparacion", help="Prefijo de las carpetas de resultados")
    parser.add_argument("--resultados", type=Path, default=RAIZ_DG / "resultados")
    parser.add_argument("--paciencia", type=int, default=10, help="Paciencia de cada corrida (la mayor del barrido)")
    parser.add_argument("--max-episodios", type=int, default=30, help="Tope de episodios de cada corrida")
    parser.add_argument("--paciencias", type=int, nargs="+", default=[3, 6, 10],
                        help="Paciencias que se calculan cortando cada corrida (<= --paciencia)")
    args = parser.parse_args(argv)
    try:
        epsilons = leer_epsilons(args.epsilons)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if max(args.paciencias) > args.paciencia:
        parser.exit(1, "--paciencias no puede tener valores mayores que --paciencia\n")
    carpetas, resumenes = barrido(epsilons, args.id, args.resultados, extra, paciencia=args.paciencia,
                                  max_episodios=args.max_episodios, paciencias=args.paciencias)
    for d in resumenes:
        print(f"epsilon {d['epsilon']}, paciencia {d['paciencia']}: C_best = {d['C_best']}, convergió en la evaluación "
              f"{(d['convergencia'] or {}).get('evaluacion')} ({d['evaluaciones']} evaluaciones)")
    print(f"Comparación: {args.resultados / (args.id + '-comparacion')}/comparacion_epsilon.csv, curvas_epsilon.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
