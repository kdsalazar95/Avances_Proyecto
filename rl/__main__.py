"""CLI del ciclo RL de la CGRA.  Uso: ``uv run python -m rl <comando> ...`` desde la raíz.

    iniciar  --config rl/config/corrida_reduccion.json      crea la corrida y evalúa la referencia
    prompt   --corrida resultados/corridas/r01                      escribe el contexto de decisión de la próxima acción
    evaluar  --corrida resultados/corridas/r01 --candidata DIR      evalúa el cgra.json que escribió el agente en DIR
    ciclo    --corrida resultados/corridas/r01 [--guion D1 D2 ...]  ciclo automático con la política de corrida.json
    resumen  --corrida resultados/corridas/r01                      muestra resumen y óptimas
    referencia DIR --computo E,V,E [--sin-este]             escribe un cgra.json de filas 'M X M' (o 'M X')
    lote     --corrida resultados/corridas/r05 DIR1 DIR2 ... [--paralelo 4]   evalúa varias configuraciones a la vez
    informe  --corrida resultados/corridas/r05                      informe de óptimas y no óptimas (informe.md)
    recalcular --corrida resultados/corridas/r04 --formula ciclos_x_nops   costos con otra fórmula, sin volver a evaluar
    gif      LOG [-o salida.gif]                            GIF del LLM trabajando, a partir de eval/<id>/skills/intento-<n>.log
    barrido  --config rl/config/comparacion.json --epsilons 0 10 20 80 90 100   una corrida por epsilon + comparación
    convergencia --corrida resultados/corridas/r01                  convergencia.csv/.json de una corrida
    comparar DIR1 DIR2 ... --salida DIR                     comparacion_epsilon.csv y curvas_epsilon.csv de varias corridas
"""

import argparse
import json
import sys
from pathlib import Path

from .agente.ciclo import Ciclo, ErrorCiclo, iniciar
from .agente.politica import PoliticaGuion, crear_politica
from .comun import ARCHIVO_METADATOS, RAIZ_REPO, leer_json, relativa
from .corrida import ErrorCorrida

CORRIDAS = RAIZ_REPO / "resultados" / "corridas"


def _mostrar(reporte):
    m = reporte["metricas"]
    print(f"{reporte['id_evaluacion']}: {reporte['estado_evaluacion']}  C = {reporte['costo_total']}  "
          f"r = {reporte['recompensa']}  C_A = {m.get('C_A')}  ciclos = {m.get('ciclos')}  "
          f"C_R = {m.get('C_R')}  nops = {m.get('nops')}")
    for op, d in (reporte.get("por_operacion") or {}).items():
        print(f"    {op:<12} {d['estado']:<16} ciclos = {d['ciclos']}  C = {d['costo']}  "
              f"(Skills 01–09: {d.get('intentos_skills')} intento(s))")
    programas = reporte.get("programas_generados") or {}
    if programas and "intentos" in programas:
        print(f"  Skills 01–09: {programas.get('intentos')} intento(s)"
              + (f", {programas['archivos']} PE*.txt" if programas.get("archivos") else ""))
    for error in reporte["errores"][:10]:
        lugar = " ".join(str(x) for x in (error.get("archivo"), error.get("ciclo") is not None and
                                          f"ciclo {error['ciclo']}" or None) if x)
        print(f"  [{error.get('skill')}/{error.get('regla')}] {lugar} {error.get('detalle')}")
    if len(reporte["errores"]) > 10:
        print(f"  ... y {len(reporte['errores']) - 10} errores más (ver reporte.json)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m rl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("iniciar")
    p.add_argument("--config", required=True)
    p.add_argument("--corridas", default=str(CORRIDAS))
    for nombre in ("prompt", "resumen"):
        sub.add_parser(nombre).add_argument("--corrida", required=True)
    p = sub.add_parser("evaluar")
    p.add_argument("--corrida", required=True)
    p.add_argument("--candidata", required=True)
    p.add_argument("--id", help="id_candidata (por defecto el de candidata.json o el siguiente it-NNNN)")
    p = sub.add_parser("ciclo")
    p.add_argument("--corrida", required=True)
    p.add_argument("--guion", nargs="+", help="carpetas con cgra.json ya escritos, en orden (política de guion)")
    p.add_argument("--max-iteraciones", type=int)
    p = sub.add_parser("referencia")
    p.add_argument("salida")
    p.add_argument("--computo", default="E,V,E", help="tipo de cómputo de cada fila (E o V)")
    p.add_argument("--sin-este", action="store_true")
    p = sub.add_parser("lote", help="evalúa varias configuraciones (carpetas con cgra.json) en paralelo")
    p.add_argument("--corrida", required=True)
    p.add_argument("candidatas", nargs="+")
    p.add_argument("--paralelo", type=int, default=4, help="configuraciones evaluadas al mismo tiempo")
    p = sub.add_parser("informe", help="escribe informe.md con las óptimas y las no óptimas de la corrida")
    p.add_argument("--corrida", required=True)
    p = sub.add_parser("recalcular", help="recalcula los costos del historial con otra fórmula de latencia")
    p.add_argument("--corrida", required=True)
    p.add_argument("--formula", default="ciclos_x_nops", help="ciclos | ciclos_x_nops")
    p = sub.add_parser("gif", help="convierte el log de los Skills 01–09 (eval/<id>/skills/intento-<n>.log) en un GIF")
    p.add_argument("log")
    p.add_argument("-o", "--salida", help="ruta del GIF (por defecto, junto al log)")
    p.add_argument("--filas", type=int, default=28, help="líneas visibles de la terminal")
    p.add_argument("--columnas", type=int, default=100, help="ancho de la terminal en caracteres")
    p = sub.add_parser("barrido", help="una corrida por epsilon (misma configuración y semilla) y su comparación")
    p.add_argument("--config", required=True)
    p.add_argument("--epsilons", nargs="+", required=True, help="0 a 1, o porcentajes: 0 10 20 80 90 100")
    p.add_argument("--corridas", default=str(CORRIDAS))
    p.add_argument("--max-iteraciones", type=int)
    sub.add_parser("convergencia").add_argument("--corrida", required=True)
    p = sub.add_parser("comparar", help="junta los datos de convergencia de varias corridas")
    p.add_argument("corridas_a_comparar", nargs="+", metavar="CORRIDA")
    p.add_argument("--salida", required=True)
    argumentos = parser.parse_args(argv)

    try:
        if argumentos.comando == "iniciar":
            ciclo = iniciar(argumentos.config, argumentos.corridas)
            print(f"Corrida creada en {relativa(ciclo.carpeta)}")
            _mostrar(ciclo.memoria.entradas[-1])
            print(f"Contexto para la próxima acción: {relativa(ciclo.escribir_prompt())}")
        elif argumentos.comando == "prompt":
            ciclo = Ciclo(argumentos.corrida)
            ruta = ciclo.escribir_prompt()
            print(ruta.read_text(encoding="utf-8"))
            print(f"(guardado en {relativa(ruta)})")
        elif argumentos.comando == "evaluar":
            ciclo = Ciclo(argumentos.corrida)
            id_candidata = argumentos.id
            metadatos = Path(argumentos.candidata) / ARCHIVO_METADATOS
            justificacion = None
            if metadatos.is_file():
                try:
                    datos = leer_json(metadatos)
                    justificacion = datos.get("justificacion")
                    propio = datos.get("id_candidata")
                except (json.JSONDecodeError, AttributeError):
                    propio = None
                if id_candidata is None and propio and propio != "referencia" and not ciclo.memoria.por_id(propio):
                    id_candidata = propio
            reporte = ciclo.evaluar_candidata(argumentos.candidata, id_candidata, justificacion)
            _mostrar(reporte)
            print(f"Mejora estricta: {reporte['_hubo_mejora']}  ·  óptimas: "
                  f"{[o['id_candidata'] for o in ciclo.memoria.reconstruir()['optimas']]}")
            print(f"Contexto para la próxima acción: {relativa(ciclo.escribir_prompt())}")
        elif argumentos.comando == "ciclo":
            ciclo = Ciclo(argumentos.corrida)
            politica = PoliticaGuion(argumentos.guion) if argumentos.guion else \
                crear_politica(ciclo.corrida["politica"], ciclo.carpeta, corrida=ciclo.corrida)
            vistos = len(ciclo.memoria.entradas)
            estado = ciclo.correr(politica, argumentos.max_iteraciones)
            for entrada in ciclo.memoria.entradas[vistos:]:
                _mostrar(entrada)
            print(f"Estado de la corrida: {estado}")
            if getattr(politica, "ultimo_prompt", None):
                print(f"Contexto para la próxima acción: {relativa(politica.ultimo_prompt)}")
        elif argumentos.comando == "resumen":
            ciclo = Ciclo(argumentos.corrida)
            print(json.dumps(ciclo.escribir_resumen(), indent=2, ensure_ascii=False))
            print(json.dumps(leer_json(ciclo.carpeta / "optimas.json"), indent=2, ensure_ascii=False))
        elif argumentos.comando == "referencia":
            from .herramientas.referencia import escribir_cgra, malla_filas
            malla = malla_filas(argumentos.computo.split(","), not argumentos.sin_este)
            print(f"Configuración escrita en {escribir_cgra(argumentos.salida, malla)}")
        elif argumentos.comando == "lote":
            from .herramientas.informe import escribir_informe
            ciclo = Ciclo(argumentos.corrida)
            justificaciones = []
            for d in argumentos.candidatas:
                try:
                    justificaciones.append(leer_json(Path(d) / ARCHIVO_METADATOS).get("justificacion"))
                except (OSError, ValueError, AttributeError):
                    justificaciones.append(None)
            for reporte in ciclo.evaluar_lote(argumentos.candidatas, justificaciones, argumentos.paralelo):
                _mostrar(reporte)
            print(f"Óptimas: {[o['id_candidata'] for o in ciclo.memoria.reconstruir()['optimas']]}")
            print(f"Informe: {relativa(escribir_informe(ciclo.carpeta))}")
            print(f"Contexto para la próxima acción: {relativa(ciclo.escribir_prompt())}")
        elif argumentos.comando == "informe":
            from .herramientas.informe import escribir_informe
            print(f"Informe: {relativa(escribir_informe(argumentos.corrida))}")
        elif argumentos.comando == "recalcular":
            from .herramientas.recalcular import recalcular
            r = recalcular(argumentos.corrida, argumentos.formula)
            print(f"Fórmula de latencia: {r['formula_latencia']}  ·  K = {r['K']}")
            for f in r["filas"]:
                malla = " / ".join(f["malla"] or [])
                costo = "—" if f["costo"] is None else f"{f['costo']:.4f}"
                original = "—" if f["costo_original"] is None else f"{f['costo_original']:.4f}"
                print(f"  {f['id']:<11} {malla:<34} {f['estado']:<16} C = {costo:<8} (antes {original})")
                for op, d in f["por_operacion"].items():
                    print(f"      {op:<12} ciclos {d['ciclos']:>5}  NOPs {d['nops']:>5} "
                          f"({100 * d['fraccion_nops']:.0f}%)  C_L {d['C_L']:.1f}  C {d['costo']:.4f}")
            print(f"Mejor: {r['mejor']}  ·  guardado en {argumentos.corrida}/recalculo_{r['formula_latencia']}.json")
        elif argumentos.comando == "gif":
            from .herramientas.gif_log import generar_gif
            try:
                ruta = generar_gif(argumentos.log, argumentos.salida, argumentos.columnas, argumentos.filas)
            except (OSError, ValueError) as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
            print(f"GIF escrito en {ruta}")
        elif argumentos.comando == "barrido":
            from .herramientas.barrido import barrido, leer_epsilons

            def avance(ciclo, estado):
                c = json.loads((ciclo.carpeta / "convergencia.json").read_text(encoding="utf-8"))
                print(f"{ciclo.corrida['id']}: {estado} · C_best = {c['C_best']} · convergió en la evaluación "
                      f"{(c['convergencia'] or {}).get('evaluacion')}", flush=True)

            try:
                epsilons = leer_epsilons(argumentos.epsilons)
            except ValueError as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
            _, resumenes = barrido(argumentos.config, epsilons, argumentos.corridas,
                                   argumentos.max_iteraciones, al_terminar=avance)
            print(f"Comparación: {len(resumenes)} corridas en {relativa(Path(argumentos.corridas))}/"
                  f"{resumenes[0]['corrida'].rsplit('-eps', 1)[0]}-comparacion/")
        elif argumentos.comando == "convergencia":
            from .herramientas.convergencia import escribir
            datos = escribir(argumentos.corrida)
            print(json.dumps(datos, indent=2, ensure_ascii=False))
        elif argumentos.comando == "comparar":
            from .herramientas.convergencia import comparar
            resumenes = comparar(argumentos.corridas_a_comparar, argumentos.salida)
            print(f"{len(resumenes)} corridas: {argumentos.salida}/comparacion_epsilon.csv, curvas_epsilon.csv")
    except (ErrorCiclo, ErrorCorrida) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
