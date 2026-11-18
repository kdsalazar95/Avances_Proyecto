"""Reporte de una corrida: tiempos, episodios, decisiones, Dijkstra, configuraciones y recursos.

Se genera al final de ``agente_greedy.py`` en ``reporte_corrida.md`` (para leer) y
``reporte_corrida.json`` (los mismos datos). Contiene:

* **Tiempos:** duración total y por fase (compilación, malla de referencia, exploración, informes) y de
  cada episodio.
* **Episodios:** cuántos, por qué terminó el ciclo y, por episodio, mallas probadas, simuladas,
  aceptadas y descartadas, malla final, costo y recompensa.
* **Decisiones del agente:** cuántas tomó Greedy y cuántas el agente LLM (válidas, inválidas,
  BAJAR/PARAR), la fracción observada frente a epsilon, y las consultas al LLM (tiempo, tiempos agotados).
* **Dijkstra y simulación:** mapeos hechos, tiempo de mapeo, validación y simulación por configuración y
  en total, fallos de mapeo y métricas del mapeo.
* **Configuraciones:** evaluadas, correctas y fallidas (por estado), eficientes (Pareto) y dominadas, y las
  de menor y mayor costo con toda su información.
* **Recursos:** tiempo de CPU (proceso y subprocesos), núcleos equivalentes y utilización, memoria pico,
  y las muestras del monitor (CPU del sistema, carga y memoria).
"""

import json
import os
import resource
import threading
import time
from datetime import datetime
from pathlib import Path

QUIEN = {"explotacion": "Greedy (explotación)", "rl": "Agente LLM", "rl_invalida": "Greedy por respuesta inválida del LLM",
         "exploracion": "Exploración al azar (sin LLM)"}


# -- recursos --------------------------------------------------------------------------------
def _memoria_mb():
    """``(total, disponible)`` en MB, de /proc/meminfo; ``(None, None)`` si no se puede leer."""
    try:
        datos = {}
        for linea in Path("/proc/meminfo").read_text().splitlines():
            clave, valor = linea.split(":", 1)
            datos[clave] = int(valor.split()[0]) / 1024
        return datos.get("MemTotal"), datos.get("MemAvailable")
    except (OSError, ValueError):
        return None, None


def _cpu_stat():
    try:
        valores = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        inactivo = valores[3] + (valores[4] if len(valores) > 4 else 0)
        return sum(valores), inactivo
    except (OSError, ValueError, IndexError):
        return None


class MonitorRecursos:
    """Toma una muestra cada ``intervalo`` segundos en un hilo: uso de CPU del sistema, carga, memoria y,
    con psutil, el árbol de procesos de la corrida (el proceso principal y todos sus descendientes: los
    procesos de las operaciones, los del mapeador y el simulador): cuántos hay, su memoria (RSS) y su
    tiempo de CPU. El CPU de un proceso se cuenta hasta su última muestra, así que los procesos que viven
    menos que el intervalo pueden quedar fuera."""

    def __init__(self, intervalo=5.0):
        self.intervalo = intervalo
        self.muestras = []
        self._cpu_por_pid = {}
        self._parar = threading.Event()
        self._hilo = None

    def _arbol(self):
        try:
            import psutil
        except ImportError:
            return None, None
        try:
            raiz = psutil.Process(os.getpid())
            procesos = [raiz] + raiz.children(recursive=True)
        except psutil.Error:
            return None, None
        rss = 0
        for proc in procesos:
            try:
                tiempos = proc.cpu_times()
                rss += proc.memory_info().rss
                clave = (proc.pid, proc.create_time())
                self._cpu_por_pid[clave] = max(self._cpu_por_pid.get(clave, 0.0), tiempos.user + tiempos.system)
            except psutil.Error:
                continue
        return len(procesos), rss / 2 ** 20

    def _muestra(self, anterior):
        actual = _cpu_stat()
        cpu = None
        if anterior and actual and actual[0] > anterior[0]:
            cpu = 100.0 * (1 - (actual[1] - anterior[1]) / (actual[0] - anterior[0]))
        total, disponible = _memoria_mb()
        try:
            carga = os.getloadavg()[0]
        except OSError:
            carga = None
        procesos, rss = self._arbol()
        self.muestras.append({"t": round(time.monotonic(), 1), "cpu_pct": cpu, "carga": carga,
                              "memoria_total_mb": total, "memoria_disponible_mb": disponible,
                              "procesos": procesos, "rss_arbol_mb": rss})
        return actual

    def _correr(self):
        anterior = _cpu_stat()
        while not self._parar.wait(self.intervalo):
            anterior = self._muestra(anterior)

    def iniciar(self):
        self._hilo = threading.Thread(target=self._correr, daemon=True)
        self._hilo.start()
        return self

    def detener(self):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=5)
        self._muestra(None)  # una última muestra del árbol de procesos
        return self.resumen()

    def resumen(self):
        def valores(clave):
            return [m[clave] for m in self.muestras if m[clave] is not None]
        cpu, carga, disp = valores("cpu_pct"), valores("carga"), valores("memoria_disponible_mb")
        procesos, rss = valores("procesos"), valores("rss_arbol_mb")
        total = next((m["memoria_total_mb"] for m in self.muestras if m["memoria_total_mb"]), _memoria_mb()[0])
        return {
            "intervalo_s": self.intervalo, "muestras": len(self.muestras),
            "cpu_sistema_promedio_pct": round(sum(cpu) / len(cpu), 1) if cpu else None,
            "cpu_sistema_max_pct": round(max(cpu), 1) if cpu else None,
            "carga_promedio": round(sum(carga) / len(carga), 2) if carga else None,
            "carga_max": round(max(carga), 2) if carga else None,
            "memoria_total_mb": round(total) if total else None,
            "memoria_disponible_min_mb": round(min(disp)) if disp else None,
            "memoria_usada_max_mb": round(total - min(disp)) if disp and total else None,
            "procesos_max": max(procesos) if procesos else None,
            "memoria_arbol_pico_mb": round(max(rss), 1) if rss else None,
            "cpu_arbol_s": round(sum(self._cpu_por_pid.values()), 1) if self._cpu_por_pid else None,
        }


def uso_cpu(duracion_s):
    """Tiempo de CPU del proceso y de sus subprocesos (mapeador, simulador), núcleos equivalentes y
    memoria pico (RSS) del proceso y del subproceso más grande."""
    yo, hijos = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = yo.ru_utime + yo.ru_stime + hijos.ru_utime + hijos.ru_stime
    nucleos = os.cpu_count()
    return {
        "nucleos_maquina": nucleos,
        "cpu_usuario_s": round(yo.ru_utime + hijos.ru_utime, 1),
        "cpu_sistema_s": round(yo.ru_stime + hijos.ru_stime, 1),
        "cpu_total_s": round(cpu, 1),
        "cpu_proceso_principal_s": round(yo.ru_utime + yo.ru_stime, 1),
        "cpu_subprocesos_s": round(hijos.ru_utime + hijos.ru_stime, 1),
        "nucleos_equivalentes": round(cpu / duracion_s, 2) if duracion_s else None,
        "utilizacion_maquina_pct": round(100 * cpu / duracion_s / nucleos, 1) if duracion_s and nucleos else None,
        "memoria_pico_proceso_mb": round(yo.ru_maxrss / 1024, 1),
        "memoria_pico_subproceso_mb": round(hijos.ru_maxrss / 1024, 1),
    }


# -- reporte ---------------------------------------------------------------------------------
def _suma(valores):
    valores = [v for v in valores if v is not None]
    return round(sum(valores), 2) if valores else None


def _resumen_configuracion(r):
    claves = ("nombre", "filas", "columnas", "pes", "n_M", "n_E", "n_V", "malla", "origen", "orden", "episodio",
              "propuesta_por", "aceptada", "estado", "motivo", "eficiente", "C", "C_A", "C_A_norm", "C_L_norm", "C_R",
              "ciclos_total", "instrucciones_total", "nops_total", "operaciones")
    return {k: r.get(k) for k in claves}


def dijkstra_y_simulacion(evaluaciones):
    """Tiempos y métricas del mapeo (Dijkstra), la validación y la simulación de las evaluaciones que los
    registraron (las hechas con esta versión del evaluador)."""
    from .tabla import nombre_malla
    por_configuracion, fallos, vistas = [], 0, set()
    for malla, e, _ in evaluaciones:
        clave = tuple(" ".join(f) if not isinstance(f, str) else f for f in malla)
        if clave in vistas:  # la referencia está en evaluacion_malla.json y también en la caché
            continue
        vistas.add(clave)
        ops = list(e.operaciones.values())
        fallos += sum(o.estado == "fallo_evaluador" for o in ops)
        if not any(o.tiempo_mapeo_s is not None for o in ops):
            continue
        por_configuracion.append({
            "configuracion": nombre_malla(clave), "estado": e.estado,
            "operaciones_mapeadas": sum(o.mapeo is not None for o in ops),
            "tiempo_mapeo_s": _suma(o.tiempo_mapeo_s for o in ops),
            "tiempo_validacion_s": _suma(o.tiempo_validacion_s for o in ops),
            "tiempo_simulacion_s": _suma(o.tiempo_simulacion_s for o in ops),
            "nodos_grafo": _suma((o.mapeo or {}).get("operaciones") for o in ops),
            "transferencias_mapeo": _suma((o.mapeo or {}).get("transferencias") for o in ops),
        })
    return {
        "configuraciones_con_tiempos": len(por_configuracion),
        "mapeos": sum(c["operaciones_mapeadas"] for c in por_configuracion),
        "fallos_de_mapeo": fallos,
        "tiempo_mapeo_total_s": _suma(c["tiempo_mapeo_s"] for c in por_configuracion),
        "tiempo_validacion_total_s": _suma(c["tiempo_validacion_s"] for c in por_configuracion),
        "tiempo_simulacion_total_s": _suma(c["tiempo_simulacion_s"] for c in por_configuracion),
        "por_configuracion": sorted(por_configuracion, key=lambda c: -(c["tiempo_mapeo_s"] or 0)),
        "nota": "Tiempos sumados sobre las operaciones; con varias operaciones en paralelo, el tiempo de reloj "
                "es menor que la suma. Las evaluaciones reusadas de corridas anteriores no tienen tiempos.",
    }


def decisiones(historial, epsilon, registro_llm):
    conteo = {}
    for ep in historial:
        for d in ep.get("decisiones", []):
            conteo[d] = conteo.get(d, 0) + 1
    control = {"BAJAR": 0, "PARAR": 0}
    for ep in historial:
        for t in ep.get("trayectoria", []):
            if t.get("accion") in control:
                control[t["accion"]] += 1
    total = sum(conteo.values())
    del_llm = conteo.get("rl", 0) + conteo.get("rl_invalida", 0)
    llm = None
    if registro_llm is not None:
        tiempos = [r["segundos"] for r in registro_llm if r["segundos"] is not None]
        llm = {
            "consultas": len(registro_llm),
            "validas": sum(bool(r["valida"]) for r in registro_llm),
            "invalidas": sum(not r["valida"] and not r["timeout"] for r in registro_llm),
            "tiempos_agotados": sum(r["timeout"] for r in registro_llm),
            "por_tipo": {t: sum(r["tipo"] == t for r in registro_llm) for t in sorted({r["tipo"] for r in registro_llm})},
            "tiempo_total_s": round(sum(tiempos), 1) if tiempos else 0,
            "tiempo_promedio_s": round(sum(tiempos) / len(tiempos), 1) if tiempos else None,
            "tiempo_max_s": round(max(tiempos), 1) if tiempos else None,
        }
    return {
        "total": total, "por_quien": {QUIEN.get(k, k): v for k, v in sorted(conteo.items())},
        "greedy": conteo.get("explotacion", 0), "llm": conteo.get("rl", 0),
        "llm_invalidas": conteo.get("rl_invalida", 0), "azar": conteo.get("exploracion", 0),
        "llm_bajar": control["BAJAR"], "llm_parar": control["PARAR"], "epsilon": epsilon,
        "fraccion_llm_observada": round(del_llm / total, 3) if total else None, "consultas_llm": llm,
    }


def _nombre(ep):
    from .tabla import nombre_malla
    return nombre_malla(ep["malla"]) if ep.get("malla") else f"{ep['filas']}x{ep['columnas']}"


def construir(parametros, fases, inicio, fin, historial, motivo_fin, c_best, registro_simulaciones, registro_llm,
              registros, evaluaciones, recursos, monitor):
    """El reporte como diccionario (ver el docstring del módulo)."""
    duracion = (fin - inicio).total_seconds()
    sims = registro_simulaciones or []
    return {
        "general": {
            "inicio": inicio.isoformat(timespec="seconds"), "fin": fin.isoformat(timespec="seconds"),
            "duracion_s": round(duracion, 1), "fases_s": {k: round(v, 1) for k, v in fases.items()},
            "modo": parametros.get("modo"), "epsilon": parametros.get("epsilon"),
            "bloques": parametros.get("bloques"), "max_episodios": parametros.get("max_episodios"),
            "paciencia": parametros.get("paciencia"), "procesos": parametros.get("procesos"),
            "mapeador_procesos": parametros.get("mapeador_procesos"), "agente_llm": parametros.get("rl"),
            "tipos": parametros.get("tipos"),
        },
        "episodios": _episodios(historial, motivo_fin, c_best),
        "decisiones": decisiones(historial, parametros.get("epsilon"), registro_llm),
        "dijkstra_y_simulacion": dijkstra_y_simulacion(evaluaciones),
        "simulaciones": {
            "en_esta_corrida": len(sims), "tiempo_total_s": _suma(s["segundos"] for s in sims),
            "mas_larga": max(sims, key=lambda s: s["segundos"]) if sims else None, "detalle": sims,
        },
        "configuraciones": _configuraciones(registros),
        "recursos": {**recursos, "monitor": monitor},
    }


def _episodios(historial, motivo_fin, c_best):
    return {
        "total": len(historial), "motivo_fin": motivo_fin, "C_best": c_best,
        "detalle": [{
            "n": ep["n"], "semilla": ep.get("semilla"), "duracion_s": ep.get("duracion_s"),
            "mallas_probadas": ep.get("simuladas"), "simuladas_ahora": ep.get("nuevas"),
            "aceptadas": ep.get("aceptadas"),
            "descartadas": (ep["simuladas"] - ep["aceptadas"]) if ep.get("simuladas") is not None else None,
            "malla_final": _nombre(ep), "estado": ep["estado"], "C": ep["C"],
            "cambios_tipo": ep.get("cambios_tipo"), "fin_tipos": ep.get("parada_tipos"),
            "recompensa": ep["recompensa"], "mejora": ep.get("mejora"), "fin": ep.get("parada"),
        } for ep in historial],
    }


def _configuraciones(registros):
    correctas = [r for r in registros if r["estado"] == "correcta"]
    por_estado = {}
    for r in registros:
        por_estado[r["estado"]] = por_estado.get(r["estado"], 0) + 1
    ordenadas = sorted(correctas, key=lambda r: r["C"])
    return {
        "evaluadas": len(registros), "correctas": len(correctas), "fallidas": len(registros) - len(correctas),
        "por_estado": por_estado, "eficientes_pareto": sum(bool(r["eficiente"]) for r in registros),
        "dominadas": sum(not r["eficiente"] for r in correctas),
        "menor_costo": _resumen_configuracion(ordenadas[0]) if ordenadas else None,
        "mayor_costo": _resumen_configuracion(ordenadas[-1]) if ordenadas else None,
        "cinco_menor_costo": [_resumen_configuracion(r) for r in ordenadas[:5]],
        "cinco_mayor_costo": [_resumen_configuracion(r) for r in ordenadas[::-1][:5]],
        "fallidas_detalle": [{"configuracion": r["nombre"], "estado": r["estado"], "motivo": r["motivo"]}
                             for r in registros if r["estado"] != "correcta"],
    }


def regenerar(carpeta):
    """Rehace ``reporte_corrida.md`` y ``.json`` de una corrida terminada **sin simular**: los tiempos,
    las decisiones y los recursos salen del ``reporte_corrida.json`` guardado; los episodios, la tabla de
    Dijkstra y las configuraciones se recalculan con los datos de la carpeta (nombres, frente de Pareto)."""
    from . import tabla
    carpeta = Path(carpeta)
    rep = json.loads((carpeta / "reporte_corrida.json").read_text(encoding="utf-8"))
    evaluaciones, episodios = tabla.de_corrida(carpeta)
    registros = tabla.construir(evaluaciones, episodios)
    datos = json.loads((carpeta / "episodios.json").read_text(encoding="utf-8"))
    rep["general"].setdefault("tipos", any(ep.get("parada_tipos") for ep in episodios or []))
    rep["episodios"] = _episodios(episodios or [], datos.get("motivo_fin", rep["episodios"].get("motivo_fin")),
                                  datos.get("C_best", rep["episodios"].get("C_best")))
    rep["dijkstra_y_simulacion"] = dijkstra_y_simulacion(evaluaciones)
    rep["configuraciones"] = _configuraciones(registros)
    escribir(carpeta, rep)
    return rep


def _n(v, d=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _duracion(s):
    if s is None:
        return "-"
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m:02d} min {s:02d} s" if h else f"{m} min {s:02d} s" if m else f"{s} s"


def _tabla_configuracion(r):
    filas = [f"- **{r['nombre']}** ({r['pes']} PEs: {r['n_M']} M, {r['n_E']} E, {r['n_V']} V), {r['estado']}, "
             f"{'eficiente (Pareto)' if r['eficiente'] else 'dominada'}",
             f"- C = {_n(r['C'], 4)} · C_A = {_n(r['C_A'])} (norm. {_n(r['C_A_norm'])}) · C_L_norm = {_n(r['C_L_norm'])} "
             f"· C_R = {_n(r['C_R'])}",
             f"- Ciclos totales {_n(r['ciclos_total'])} · instrucciones {_n(r['instrucciones_total'])} · NOPs {_n(r['nops_total'])}",
             f"- Exploración: orden {_n(r['orden'])}, episodio {_n(r['episodio'])}, propuesta por "
             f"{QUIEN.get(r['propuesta_por'], r['propuesta_por']) or '-'}, aceptada {_n(r['aceptada'])}", ""]
    if r["pes"] <= 64:
        filas += ["Distribución de los PEs (fila 0 arriba; M = memoria, E = escalar, V = vectorial):", "", "```"]
        filas += r["malla"] + ["```", ""]
    filas += [
             "| Operación | Ciclos | Cómputo | Comunic. | Inactivos | Instrucciones | NOPs | Lecturas | Escrituras | "
             "Transf. | C_L_norm | C_R | Costo |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for op, d in r["operaciones"].items():
        filas.append(f"| {op} | {_n(d['ciclos'])} | {_n(d['ciclos_computo'])} | {_n(d['ciclos_comunicacion'])} | "
                     f"{_n(d['ciclos_inactivos'])} | {_n(d['instrucciones'])} | {_n(d['nops'])} | {_n(d['lecturas'])} | "
                     f"{_n(d['escrituras'])} | {_n(d['transferencias'])} | {_n(d['C_L_norm'])} | {_n(d['C_R'])} | "
                     f"{_n(d['costo'])} |")
    return "\n".join(filas)


def markdown(rep):
    g, ep, dec, dj = rep["general"], rep["episodios"], rep["decisiones"], rep["dijkstra_y_simulacion"]
    sim, conf, rec = rep["simulaciones"], rep["configuraciones"], rep["recursos"]
    mon = rec.get("monitor") or {}
    l = ["# Reporte de la corrida", "",
         f"Modo **{g['modo']}** · epsilon {g['epsilon']} · bloques {g['bloques']} · "
         f"máx. {g['max_episodios']} episodios, paciencia {g['paciencia']}"
         + (f" · fase de tipos (E/V por PE): {'sí' if g.get('tipos') else 'no'}" if g.get("modo") == "bloques" else "")
         + f" · procesos del simulador: {g['procesos'] or 'uno por operación'} · procesos del mapeador: "
         f"{g['mapeador_procesos'] or 1}", "",
         "## Tiempo", "", f"- Inicio: {g['inicio']} · fin: {g['fin']}", f"- **Duración total: {_duracion(g['duracion_s'])}**"]
    l += [f"- {fase}: {_duracion(s)}" for fase, s in g["fases_s"].items()]
    l += ["", "## Episodios", "", f"{ep['total']} episodios, fin por **{ep['motivo_fin']}**; mejor C = {_n(ep['C_best'], 4)}", "",
          "Malla final: tamaño y, en las pequeñas, la distribución de tipos del interior fila por fila entre las "
          "columnas M (por ejemplo `2x4 [VE/VV]` = `M V E M / M V V M`).", "",
          "| Episodio | Duración | Probadas | Simuladas ahora | Aceptadas | Descartadas | Cambios E/V | Malla final | C | R | "
          "¿Mejora? | Fin (descenso / tipos) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    l += [f"| {e['n']} | {_duracion(e['duracion_s'])} | {_n(e['mallas_probadas'])} | {_n(e['simuladas_ahora'])} | "
          f"{_n(e['aceptadas'])} | {_n(e['descartadas'])} | {_n(e.get('cambios_tipo'))} | {e['malla_final']} | "
          f"{_n(e['C'], 4)} | {_n(e['recompensa'], 4)} | {'sí' if e['mejora'] else 'no'} | {e['fin']}"
          + (f" / {e['fin_tipos']}" if e.get("fin_tipos") else "") + " |" for e in ep["detalle"]]
    l += ["", "## Decisiones del agente", "",
          f"{dec['total']} decisiones · fracción del LLM observada {_n(dec['fraccion_llm_observada'])} (epsilon = {dec['epsilon']})", "",
          "| Quién decidió | Decisiones |", "|---|---|"]
    l += [f"| {k} | {v} |" for k, v in dec["por_quien"].items()]
    l += [f"| Agente LLM: BAJAR de bloque | {dec['llm_bajar']} |", f"| Agente LLM: PARAR | {dec['llm_parar']} |"]
    if dec["consultas_llm"]:
        c = dec["consultas_llm"]
        l += ["", f"Consultas al LLM: {c['consultas']} ({c['validas']} válidas, {c['invalidas']} inválidas, "
                  f"{c['tiempos_agotados']} tiempos agotados) · tiempo total {_duracion(c['tiempo_total_s'])}, "
                  f"promedio {_n(c['tiempo_promedio_s'], 1)} s, máximo {_n(c['tiempo_max_s'], 1)} s"]
    l += ["", "## Dijkstra y simulación", "",
          f"- Configuraciones con tiempos: {dj['configuraciones_con_tiempos']} · mapeos con Dijkstra: {dj['mapeos']} · "
          f"fallos de mapeo: {dj['fallos_de_mapeo']}",
          f"- Tiempo de mapeo (Dijkstra): {_duracion(dj['tiempo_mapeo_total_s'])} · validación E03: "
          f"{_duracion(dj['tiempo_validacion_total_s'])} · simulación: {_duracion(dj['tiempo_simulacion_total_s'])}",
          f"- Simulaciones en esta corrida: {sim['en_esta_corrida']} (tiempo de reloj {_duracion(sim['tiempo_total_s'])})"
          + (f"; la más larga: {sim['mas_larga']['configuracion']} en {_duracion(sim['mas_larga']['segundos'])}"
             if sim["mas_larga"] else ""), f"- {dj['nota']}", "",
          "| Configuración | Mapeo (Dijkstra) | Validación | Simulación | Operaciones del grafo de datos | Transferencias |",
          "|---|---|---|---|---|---|"]
    l += [f"| {c['configuracion']} | {_duracion(c['tiempo_mapeo_s'])} | {_duracion(c['tiempo_validacion_s'])} | "
          f"{_duracion(c['tiempo_simulacion_s'])} | {_n(c['nodos_grafo'])} | {_n(c['transferencias_mapeo'])} |"
          for c in dj["por_configuracion"]]
    l += ["", "## Configuraciones", "",
          f"- Evaluadas: **{conf['evaluadas']}** · correctas: **{conf['correctas']}** · fallidas: {conf['fallidas']} "
          f"({', '.join(f'{k}: {v}' for k, v in conf['por_estado'].items())})",
          f"- Eficientes (frente de Pareto): {conf['eficientes_pareto']} · dominadas: {conf['dominadas']}", ""]
    if conf["menor_costo"]:
        l += ["### Configuración de menor costo", "", _tabla_configuracion(conf["menor_costo"]), "",
              "### Configuración de mayor costo", "", _tabla_configuracion(conf["mayor_costo"]), "",
              "### Las cinco de menor y mayor costo", "", "| | Configuración | PEs | C | C_A | C_L_norm | Pareto |",
              "|---|---|---|---|---|---|---|"]
        l += [f"| menor {i + 1} | {r['nombre']} | {r['pes']} | {_n(r['C'], 4)} | {_n(r['C_A'])} | {_n(r['C_L_norm'])} | "
              f"{'sí' if r['eficiente'] else 'no'} |" for i, r in enumerate(conf["cinco_menor_costo"])]
        l += [f"| mayor {i + 1} | {r['nombre']} | {r['pes']} | {_n(r['C'], 4)} | {_n(r['C_A'])} | {_n(r['C_L_norm'])} | "
              f"{'sí' if r['eficiente'] else 'no'} |" for i, r in enumerate(conf["cinco_mayor_costo"])]
    if conf["fallidas_detalle"]:
        l += ["", "### Configuraciones fallidas", "", "| Configuración | Estado | Motivo |", "|---|---|---|"]
        l += [f"| {f['configuracion']} | {f['estado']} | {(f['motivo'] or '')[:120]} |" for f in conf["fallidas_detalle"]]
    cpu_arbol = mon.get("cpu_arbol_s")
    l += ["", "## Recursos computacionales", "",
          f"- Núcleos de la máquina: {rec['nucleos_maquina']}",
          f"- Tiempo de CPU de la corrida (proceso principal, operaciones, mapeador y simulador): "
          f"{_duracion(cpu_arbol if cpu_arbol is not None else rec['cpu_total_s'])}"
          + (f" · {_n(cpu_arbol / g['duracion_s'], 2)} núcleos en promedio · utilización media de la máquina "
             f"{_n(100 * cpu_arbol / g['duracion_s'] / rec['nucleos_maquina'], 1)} %"
             if cpu_arbol and g["duracion_s"] and rec.get("nucleos_maquina") else ""),
          f"- Procesos de la corrida a la vez: hasta {_n(mon.get('procesos_max'))} · memoria pico del conjunto: "
          f"{_n(mon.get('memoria_arbol_pico_mb'), 1)} MB",
          f"- Monitor ({mon.get('muestras', 0)} muestras cada {mon.get('intervalo_s')} s): CPU del sistema promedio "
          f"{_n(mon.get('cpu_sistema_promedio_pct'), 1)} %, máx. {_n(mon.get('cpu_sistema_max_pct'), 1)} % · carga promedio "
          f"{_n(mon.get('carga_promedio'), 2)}, máx. {_n(mon.get('carga_max'), 2)} · memoria usada máx. "
          f"{_n(mon.get('memoria_usada_max_mb'))} MB de {_n(mon.get('memoria_total_mb'))} MB "
          f"(disponible mín. {_n(mon.get('memoria_disponible_min_mb'))} MB)",
          "- Medido con un monitor del árbol de procesos (psutil) cada pocos segundos; getrusage no sirve aquí "
          "porque no ve los procesos del mapeador mientras corren.", ""]
    return "\n".join(l)


def escribir(salida, rep):
    salida = Path(salida)
    (salida / "reporte_corrida.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, allow_nan=False,
                                                            default=str) + "\n", encoding="utf-8")
    (salida / "reporte_corrida.md").write_text(markdown(rep), encoding="utf-8")


def ahora():
    return datetime.now().astimezone()
