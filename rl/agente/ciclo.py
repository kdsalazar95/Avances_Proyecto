"""A08 · Control de iteraciones del ciclo RL.

    corridas/<id>/
    ├── corrida.json        parámetros congelados de la corrida
    ├── contexto.json       A01
    ├── instancia/          memoria.bin, esperado.json (privado del evaluador), instancia.json
    ├── normalizacion.json  C_A,ref y C_L,ref: los fijos de corrida.json ("normalizadores") o los
    │                       medidos en la referencia
    ├── episodios.json      con "max_episodios" en los límites: los episodios terminados (agente/episodios.py)
    ├── convergencia.csv/.json  costo y tiempo de cada evaluación y el resumen de convergencia
    ├── candidatas/<id>/    acciones del agente: cgra.json + candidata.json (solo lectura)
    ├── eval/<id>/          evaluación aislada: skills/intento-k/ (Skills 01–09), PE*.txt,
    │                       reporte_ciclos.txt, memoria final y reporte.json
    ├── prompts/<id>.md     contexto de decisión entregado a la política
    ├── historial.jsonl     una línea por evaluación (solo agregar)
    ├── optimas.json        todas las correctas empatadas en el menor costo
    └── resumen.json        estado de la corrida

El agente y el evaluador se llaman por sus interfaces; el ciclo solo usa el
reporte E08 que devuelve ``evaluar``.
"""

import copy
import json
import shutil
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import estados
from ..comun import ARCHIVO_CONFIG, ahora, escribir_json, leer_json, relativa
from ..corrida import ErrorCorrida, cargar_corrida, es_conjunto, normalizadores_fijos, resolver, validar_corrida
from ..entorno.contexto import cargar_contexto, cargar_contexto_conjunto
from ..entorno.validacion_accion import validar_accion
from ..evaluador import reporte as e08
from ..evaluador.entradas import preparar_instancia
from ..evaluador.compilador import crear_compilador
from ..evaluador.conjunto import evaluar_conjunto
from ..evaluador.evaluacion import accion_de, evaluar
from ..evaluador.recompensa import costo_reportado, recompensa, verificar_penalizacion
from .episodios import CacheEvaluaciones, clave_hardware, estado_episodio
from .memoria import Memoria
from .optimas import actualizar_optimas
from .politica import PoliticaManual
from .recepcion import recibir_reporte

ID_REFERENCIA = "referencia"


class ErrorCiclo(RuntimeError):
    pass


def _solo_lectura(directorio):
    for ruta in Path(directorio).rglob("*"):
        if ruta.is_file():
            ruta.chmod(ruta.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _malla(directorio):
    try:
        datos = leer_json(Path(directorio) / ARCHIVO_CONFIG)
        return datos.get("malla")
    except Exception:  # noqa: BLE001 - un cgra.json ilegible ya lo reporta A03
        return None


def iniciar(ruta_config, carpeta_corridas, cambios=None, referencia_de=None):
    """Crea la corrida, prepara la instancia, evalúa la referencia y devuelve el ``Ciclo``.

    ``cambios``: valores que reemplazan los de la configuración (p. ej. ``id`` y ``epsilon`` en un barrido).
    ``referencia_de``: carpeta de otra corrida con la misma referencia y los mismos normalizadores; su
    evaluación de la referencia se reusa en lugar de repetirla (la 32x32 con los Skills es lo más caro).
    """
    corrida = cargar_corrida(ruta_config)
    if cambios:
        corrida.update(cambios)
        errores = validar_corrida(corrida)
        if errores:
            raise ErrorCorrida("; ".join(errores))
    fijos = normalizadores_fijos(corrida)
    carpeta = Path(carpeta_corridas) / corrida["id"]
    if carpeta.exists():
        raise ErrorCiclo(f"La corrida {carpeta} ya existe; use 'reanudar' o cambie el id")
    carpeta.mkdir(parents=True)
    corrida["creada"] = ahora()
    corrida["config_origen"] = relativa(ruta_config)
    escribir_json(carpeta / "corrida.json", corrida)

    if es_conjunto(corrida):
        # Una instancia por operación: la configuración se evalúa con todas.
        instancias = {op: preparar_instancia(resolver(corrida["fuente_c"]), op, carpeta / "instancia" / op)
                      for op in corrida["operaciones"]}
        contexto = cargar_contexto_conjunto(corrida, instancias)
    else:
        instancia = preparar_instancia(resolver(corrida["fuente_c"]), corrida["operacion"], carpeta / "instancia")
        contexto = cargar_contexto(corrida, instancia)
    escribir_json(carpeta / "contexto.json", contexto)
    if fijos:
        escribir_json(carpeta / "normalizacion.json", fijos)

    # La referencia es una configuración (cgra.json); pasa por el mismo evaluador (Skills 01–09 + simulador).
    destino = carpeta / "candidatas" / ID_REFERENCIA
    referencia = corrida["referencia"]
    if "cgra" in referencia:
        destino.mkdir(parents=True)
        shutil.copy2(resolver(referencia["cgra"]), destino / ARCHIVO_CONFIG)
    else:
        from ..herramientas.referencia import escribir_cgra
        escribir_cgra(destino, referencia["generar"]["malla"], esquema=resolver(corrida["esquema_cgra"]),
                      id_candidata=ID_REFERENCIA, justificacion="configuración de referencia de la corrida")
    _solo_lectura(destino)

    ciclo = Ciclo(carpeta)
    reporte = ciclo.reusar_referencia(referencia_de) if referencia_de else ciclo.evaluar_referencia()
    if reporte["estado_evaluacion"] != estados.CORRECTA:
        ciclo.escribir_resumen("detenida: referencia no válida")
        raise ErrorCiclo(f"La referencia no es correcta ({reporte['estado_evaluacion']}): "
                         f"{[e.get('detalle') for e in reporte['errores']][:5]}")
    return ciclo


class Ciclo:
    def __init__(self, carpeta):
        self.carpeta = Path(carpeta).resolve()  # absoluta: el simulador y los skills corren en otras carpetas
        if not (self.carpeta / "corrida.json").is_file():
            raise ErrorCiclo(f"{self.carpeta} no es una corrida (falta corrida.json)")
        self.corrida = leer_json(self.carpeta / "corrida.json")
        self.contexto = leer_json(self.carpeta / "contexto.json")
        if es_conjunto(self.corrida):
            self.instancia = None
            self.instancias = {op: leer_json(self.carpeta / "instancia" / op / "instancia.json")
                               for op in self.corrida["operaciones"]}
        else:
            self.instancia = leer_json(self.carpeta / "instancia" / "instancia.json")
            self.instancias = None
        self.memoria = Memoria(self.carpeta / "historial.jsonl", self.corrida["tolerancia_empate_costo"])
        ruta_norm = self.carpeta / "normalizacion.json"
        self.normalizadores = leer_json(ruta_norm) if ruta_norm.is_file() else None
        self.compilador = crear_compilador(self.corrida["compilador"])
        self.estado_corrida = None
        cache = self.corrida.get("cache_evaluaciones")
        self.cache = CacheEvaluaciones(resolver(cache)) if cache else None

    # --- referencia -------------------------------------------------------
    def evaluar_referencia(self):
        if self.memoria.por_id(ID_REFERENCIA):
            raise ErrorCiclo("la referencia ya fue evaluada en esta corrida")
        directorio = self.carpeta / "candidatas" / ID_REFERENCIA
        fijos = bool(self.normalizadores and self.normalizadores.get("fijos"))
        reporte = self._evaluar(directorio, ID_REFERENCIA, normalizadores=self.normalizadores if fijos else None,
                                mejor_anterior=None, es_referencia=True)
        if not fijos and reporte["estado_evaluacion"] == estados.CORRECTA:
            self.normalizadores = reporte["parametros"]["normalizadores"]
            escribir_json(self.carpeta / "normalizacion.json", self.normalizadores)
        reporte["decidio"] = ID_REFERENCIA
        self._guardar_en_cache(directorio, reporte)
        self._registrar(reporte, directorio, iteracion=0)
        return reporte

    def reusar_referencia(self, origen):
        """Registra la evaluación de la referencia de otra corrida (misma malla y mismos normalizadores)."""
        if self.memoria.por_id(ID_REFERENCIA):
            raise ErrorCiclo("la referencia ya fue evaluada en esta corrida")
        origen = Path(origen).resolve()
        previa = Memoria(origen / "historial.jsonl").por_id(ID_REFERENCIA)
        if previa is None or previa["estado_evaluacion"] != estados.CORRECTA:
            raise ErrorCiclo(f"{origen} no tiene una referencia correcta para reusar")
        directorio = self.carpeta / "candidatas" / ID_REFERENCIA
        if _malla(directorio) != _malla(origen / "candidatas" / ID_REFERENCIA):
            raise ErrorCiclo(f"la referencia de {origen} es otra malla: no se puede reusar")
        normalizadores_origen = leer_json(origen / "normalizacion.json")
        if self.normalizadores is None:
            self.normalizadores = normalizadores_origen
            escribir_json(self.carpeta / "normalizacion.json", self.normalizadores)
        elif self.normalizadores != normalizadores_origen:
            raise ErrorCiclo(f"{origen} usa otros normalizadores: no se puede reusar su referencia")
        if (origen / "eval" / ID_REFERENCIA).is_dir():
            shutil.copytree(origen / "eval" / ID_REFERENCIA, self.carpeta / "eval" / ID_REFERENCIA, symlinks=True)
        reporte = {k: v for k, v in previa.items()
                   if k not in ("iteracion", "marca_tiempo", "duplicada_de", "malla", "recibido_A04")}
        reporte["reusada_de"] = relativa(origen)
        reporte["clave_hardware"] = clave_hardware(directorio)
        self._registrar(reporte, directorio, iteracion=0)
        return reporte

    # --- una iteración ----------------------------------------------------
    def siguiente_id(self):
        return f"it-{self.memoria.siguiente_iteracion():04d}"

    def _apartar_interrumpidos(self, id_candidata):
        """Una candidata/evaluación sin entrada en el historial viene de una corrida interrumpida."""
        for base in ("candidatas", "eval"):
            ruta = self.carpeta / base / id_candidata
            if ruta.exists():
                ruta.rename(ruta.with_name(f"{id_candidata}.interrumpida-{int(time.time())}"))

    def registrar_candidata(self, origen, id_candidata):
        """Copia la acción del agente a candidatas/<id>/ y la deja de solo lectura."""
        destino = self.carpeta / "candidatas" / id_candidata
        origen = Path(origen)
        if origen.resolve() != destino.resolve():
            if destino.exists():
                raise ErrorCiclo(f"{destino} ya existe: las candidatas no se sobrescriben")
            shutil.copytree(origen, destino, symlinks=True)
        _solo_lectura(destino)
        return destino

    def evaluar_candidata(self, origen, id_candidata=None, justificacion=None, decidio="llm", tiempo_decision_s=None,
                          mejor_anterior=None, episodio=None):
        """A03 → evaluador (E01–E08) → A04 → A05 → A07 para un cgra.json ya escrito por el agente.

        ``decidio``: quién propuso la configuración (``llm`` o ``exploracion``, la rama al azar de epsilon);
        ``tiempo_decision_s``: lo que tardó en proponerla (para los tiempos de convergencia).
        ``mejor_anterior``: el costo contra el que se mide la recompensa (por defecto, el mejor de la corrida; en
        un episodio, el mejor del episodio). ``episodio``: ``{"n", "paso"}``. Una configuración ya evaluada (en
        esta corrida o en la caché compartida) no se vuelve a evaluar (``_de_cache``)."""
        if self.normalizadores is None:
            raise ErrorCiclo("no hay referencia correcta evaluada: no se puede optimizar")
        id_candidata = id_candidata or self.siguiente_id()
        if self.memoria.por_id(id_candidata):
            raise ErrorCiclo(f"'{id_candidata}' ya fue evaluada")
        directorio = self.registrar_candidata(origen, id_candidata)
        if mejor_anterior is None:
            mejor_anterior = self.memoria.mejor_costo()
        reporte = self._de_cache(directorio, id_candidata, mejor_anterior)
        if reporte is None:
            reporte = self._evaluar(directorio, id_candidata, self.normalizadores, mejor_anterior)
            self._guardar_en_cache(directorio, reporte)
        if justificacion:
            reporte["justificacion_agente"] = justificacion
        reporte["decidio"] = decidio
        reporte["tiempo_decision_s"] = tiempo_decision_s
        if episodio is not None:
            reporte["episodio"] = episodio
        return self._registrar(reporte, directorio, iteracion=self.memoria.siguiente_iteracion())

    def evaluar_lote(self, origenes, justificaciones=None, paralelo=4):
        """Evalúa varias configuraciones a la vez (un lote del agente).

        Las candidatas reciben ids seguidos y se evalúan en paralelo (cada una en su eval/<id>/); después
        se registran en orden. La recompensa de todas se mide contra el mejor costo **antes** del lote.
        """
        if self.normalizadores is None:
            raise ErrorCiclo("no hay referencia correcta evaluada: no se puede optimizar")
        justificaciones = justificaciones or [None] * len(origenes)
        primera = self.memoria.siguiente_iteracion()
        ids = [f"it-{primera + k:04d}" for k in range(len(origenes))]
        directorios = []
        for origen, id_candidata in zip(origenes, ids):
            self._apartar_interrumpidos(id_candidata)
            directorios.append(self.registrar_candidata(origen, id_candidata))
        mejor_anterior = self.memoria.mejor_costo()

        def una(k):
            return self._evaluar(directorios[k], ids[k], self.normalizadores, mejor_anterior)

        with ThreadPoolExecutor(max_workers=max(1, paralelo)) as hilos:
            reportes = list(hilos.map(una, range(len(ids))))
        registrados = []
        for k, reporte in enumerate(reportes):
            if justificaciones[k]:
                reporte["justificacion_agente"] = justificaciones[k]
            reporte["decidio"] = "llm"
            registrados.append(self._registrar(reporte, directorios[k], iteracion=primera + k))
        return registrados

    # --- caché de evaluaciones ---------------------------------------------
    QUITAR_DE_CACHE = ("iteracion", "marca_tiempo", "duplicada_de", "malla", "recibido_A04", "_hubo_mejora",
                       "justificacion_agente", "decidio", "tiempo_decision_s", "episodio", "reusada_de", "repetida",
                       "es_referencia")

    def _guardar_en_cache(self, directorio, reporte):
        """Anota la clave de hardware en el reporte y lo guarda en la caché compartida (si la hay)."""
        if reporte["estado_evaluacion"] == estados.ACCION_INVALIDA:
            return
        try:
            reporte["clave_hardware"] = clave_hardware(directorio)
        except (OSError, ValueError):
            return
        if self.cache is not None:
            limpio = {k: v for k, v in reporte.items() if k not in self.QUITAR_DE_CACHE}
            self.cache.guardar(reporte["clave_hardware"], limpio, self.corrida["id"])

    def _de_cache(self, directorio, id_candidata, mejor_anterior):
        """El reporte de una configuración ya evaluada (misma clave de hardware), con el id y la recompensa de esta
        acción, o None. ``repetida``: ya salió en esta corrida (no costó tiempo); si no, viene de la caché compartida
        y conserva el tiempo que tardó cuando se evaluó."""
        if not validar_accion(directorio, self.contexto, id_candidata)["paquete_valido"]:
            return None
        clave = clave_hardware(directorio)
        previa = next((e for e in self.memoria.entradas if e.get("clave_hardware") == clave
                       and e["estado_evaluacion"] not in estados.NO_ATRIBUIBLES), None)
        base = previa if previa is not None else (self.cache.buscar(clave) if self.cache else None)
        if base is None:
            return None
        reporte = copy.deepcopy({k: v for k, v in base.items() if k not in self.QUITAR_DE_CACHE})
        reporte["id_evaluacion"] = id_candidata
        reporte["accion"] = accion_de(id_candidata, directorio)
        p_fallo = self.corrida["penalizacion"]["P_fallo"]
        reporte["recompensa"], reporte["detalle_recompensa"] = recompensa(reporte["estado_evaluacion"],
                                                                         reporte["costo_total"], mejor_anterior,
                                                                         p_fallo)
        reporte["es_referencia"] = False
        reporte["clave_hardware"] = clave
        reporte["repetida"] = previa is not None
        reporte["reusada_de"] = previa["id_evaluacion"] if previa is not None else "cache"
        escribir_json(self.carpeta / "eval" / id_candidata / "reporte.json", reporte)
        return reporte

    def _evaluar(self, directorio, id_candidata, normalizadores, mejor_anterior, es_referencia=None):
        validacion = validar_accion(directorio, self.contexto, id_candidata)
        carpeta_eval = self.carpeta / "eval" / id_candidata
        if not validacion["paquete_valido"]:
            return self._reporte_invalido(directorio, id_candidata, validacion["motivos"], normalizadores,
                                          mejor_anterior, carpeta_eval)
        if self.instancias is not None:
            return evaluar_conjunto(directorio, id_candidata, self.instancias, self.corrida, normalizadores,
                                    mejor_anterior, carpeta_eval, self.compilador, es_referencia=es_referencia)
        return evaluar(directorio, id_candidata, self.instancia, self.corrida, normalizadores, mejor_anterior,
                       carpeta_eval, self.compilador, es_referencia=es_referencia)

    def _reporte_invalido(self, directorio, id_candidata, motivos, normalizadores, mejor_anterior, carpeta_eval):
        """A03 falló: no se llama al evaluador; se registra accion_invalida con sus motivos."""
        p_fallo = self.corrida["penalizacion"]["P_fallo"]
        accion = accion_de(id_candidata, directorio)
        r, detalle = recompensa(estados.ACCION_INVALIDA, p_fallo, mejor_anterior, p_fallo)
        doc = e08.construir(id_candidata, accion, estados.ACCION_INVALIDA,
                            costo_total=costo_reportado(estados.ACCION_INVALIDA, None, p_fallo), recompensa=r,
                            errores=[{"skill": "A03", "regla": "paquete", "detalle": m} for m in motivos],
                            tiempo_s=0.0, parametros=e08.parametros_usados(self.corrida, normalizadores),
                            detalle_recompensa=detalle)
        doc["es_referencia"] = id_candidata == ID_REFERENCIA
        escribir_json(Path(carpeta_eval) / "reporte.json", doc)
        return doc

    def _registrar(self, reporte, directorio, iteracion):
        recibido = recibir_reporte(reporte, reporte["id_evaluacion"], directorio)        # A04
        previas = self.memoria.reconstruir()
        self.memoria.agregar(reporte, iteracion, recibido, _malla(directorio))            # A05
        optimas, c_min, mejora = actualizar_optimas(previas["optimas"], reporte,          # A07
                                                    self.corrida["tolerancia_empate_costo"])
        escribir_json(self.carpeta / "optimas.json", {"C_min": c_min, "optimas": optimas,
                                                      "tolerancia_empate_costo":
                                                          self.corrida["tolerancia_empate_costo"]})
        reporte["_hubo_mejora"] = mejora
        self.escribir_resumen()
        return reporte

    def escribir_prompt(self):
        """Escribe en prompts/<id>.md la entrada π(C, E) de la próxima acción y devuelve su ruta."""
        politica = PoliticaManual(self.carpeta / "prompts")
        id_candidata = self.siguiente_id()
        politica.proponer(self.contexto, self.memoria.entrada_politica(), id_candidata,
                          self.carpeta / "candidatas" / id_candidata)
        return politica.ultimo_prompt

    # --- ciclo completo ---------------------------------------------------
    def correr(self, politica, max_iteraciones=None):
        """Repite proponer → validar → evaluar → registrar hasta un criterio de fin. Con ``max_episodios`` en los
        límites, por episodios (``correr_episodios``); si no, por iteraciones."""
        if self.corrida["limites"].get("max_episodios"):
            return self.correr_episodios(politica)
        limites = self.corrida["limites"]
        max_iteraciones = max_iteraciones or limites["max_iteraciones"]
        inicio = time.monotonic()
        while True:
            hechas = len([e for e in self.memoria.entradas if not e.get("es_referencia")])
            estado = self.memoria.reconstruir()
            if hechas >= max_iteraciones:
                self.estado_corrida = "terminada: max_iteraciones"
                break
            if estado["sin_mejora"] >= limites["paciencia"]:
                self.estado_corrida = "terminada: paciencia"
                break
            if limites.get("t_max_total_s") and time.monotonic() - inicio >= limites["t_max_total_s"]:
                self.estado_corrida = "terminada: t_max_total"
                break
            id_candidata = self.siguiente_id()
            self._apartar_interrumpidos(id_candidata)
            dir_salida = self.carpeta / "candidatas" / id_candidata
            t_decision = time.monotonic()
            propuesta = politica.proponer(self.contexto, self.memoria.entrada_politica(),
                                          id_candidata, dir_salida)    # A06/A02: π(C, E) = A
            t_decision = round(time.monotonic() - t_decision, 3)
            if propuesta is None:
                self.estado_corrida = "en_pausa: esperando acción del agente"
                break
            if not dir_salida.exists():
                dir_salida.mkdir(parents=True)  # la política no entregó nada: A03 la marcará inválida
            reporte = self.evaluar_candidata(dir_salida, id_candidata, propuesta.get("justificacion"),
                                             propuesta.get("decidio", "llm"), t_decision)
            politica.incorporar(recibir_reporte(reporte, id_candidata, dir_salida))
        self.escribir_resumen(self.estado_corrida)
        from ..herramientas.convergencia import escribir as escribir_convergencia
        escribir_convergencia(self.carpeta)
        return self.estado_corrida

    # --- episodios ---------------------------------------------------------
    def episodios(self):
        ruta = self.carpeta / "episodios.json"
        return leer_json(ruta)["episodios"] if ruta.is_file() else []

    def _escribir_episodios(self, episodios, motivo_fin=None):
        limites = self.corrida["limites"]
        referencia = self.memoria.por_id(ID_REFERENCIA)
        mejores = [ep["C"] for ep in episodios if ep.get("C") is not None]
        escribir_json(self.carpeta / "episodios.json", {
            "max_episodios": limites["max_episodios"], "paciencia": limites["paciencia"],
            "sin_mejora_episodio": limites.get("sin_mejora_episodio", 3),
            "max_pasos_episodio": limites.get("max_pasos_episodio", 40),
            "epsilon": self.corrida.get("epsilon", 0.0), "semilla": self.corrida.get("semilla", 0),
            "C_ref": referencia["costo_total"] if referencia else None,
            "C_best": min(mejores) if mejores else None, "motivo_fin": motivo_fin, "episodios": episodios,
        })

    def _entrada_episodio(self, n, entradas_ep, estado, anteriores):
        """La entrada de la política en un episodio: la de ``Memoria.entrada_politica`` (en el primer paso, sin E: el
        episodio empieza en la referencia) más el episodio (su mejor, cuántas acciones no la mejoraron y los
        episodios anteriores con su mejor configuración)."""
        entrada = self.memoria.entrada_politica() or {}
        if not entradas_ep:
            entrada = {**entrada, "E": None}
        mejor = estado["mejor"]
        entrada["episodio"] = {
            "n": n, "paso": estado["pasos"],
            "mejor": {**Memoria._costos(mejor), "malla": mejor.get("malla")},
            "sin_mejora": estado["sin_mejora"],
            "limite_sin_mejora": self.corrida["limites"].get("sin_mejora_episodio", 3),
            "anteriores": [{"n": ep["n"], "malla": ep["malla"], "C": ep["C"], "mejora": ep["mejora"]}
                           for ep in anteriores],
        }
        return entrada

    def correr_episodios(self, politica):
        """El ciclo por episodios (``agente/episodios.py``), con la regla de paciencia del Sistema 2. Se puede
        reanudar: los episodios terminados están en ``episodios.json`` y el que estaba en curso se reconstruye con
        sus evaluaciones del historial."""
        limites = self.corrida["limites"]
        tolerancia = self.corrida["tolerancia_empate_costo"]
        sin_mejora_max = limites.get("sin_mejora_episodio", 3)
        max_pasos = limites.get("max_pasos_episodio", 40)
        referencia = self.memoria.por_id(ID_REFERENCIA)
        if referencia is None or referencia["estado_evaluacion"] != estados.CORRECTA:
            raise ErrorCiclo("no hay referencia correcta evaluada: no se puede optimizar")
        inicio = time.monotonic()
        while True:
            episodios = self.episodios()
            c_best, sin_mejora = referencia["costo_total"], 0
            for ep in episodios:
                if ep["mejora"]:
                    c_best, sin_mejora = ep["C"], 0
                elif ep["estado"] not in estados.NO_ATRIBUIBLES:
                    sin_mejora += 1
            if len(episodios) >= limites["max_episodios"]:
                self.estado_corrida = "terminada: max_episodios"
                break
            if sin_mejora >= limites["paciencia"]:
                self.estado_corrida = "terminada: paciencia"
                break
            if limites.get("t_max_total_s") and time.monotonic() - inicio >= limites["t_max_total_s"]:
                self.estado_corrida = "terminada: t_max_total"
                break
            n = len(episodios)
            motivo, pausa = None, False
            while motivo is None:
                entradas_ep = [e for e in self.memoria.entradas if (e.get("episodio") or {}).get("n") == n]
                estado = estado_episodio(entradas_ep, referencia, tolerancia)
                if estado["sin_mejora"] >= sin_mejora_max:
                    motivo = "sin_mejora"
                    break
                if estado["pasos"] >= max_pasos:
                    motivo = "tope"
                    break
                id_candidata = self.siguiente_id()
                self._apartar_interrumpidos(id_candidata)
                dir_salida = self.carpeta / "candidatas" / id_candidata
                t_decision = time.monotonic()
                propuesta = politica.proponer(self.contexto, self._entrada_episodio(n, entradas_ep, estado, episodios),
                                              id_candidata, dir_salida)
                t_decision = round(time.monotonic() - t_decision, 3)
                if propuesta is None:
                    pausa = True
                    break
                if propuesta.get("parar"):
                    dir_salida.rename(dir_salida.with_name(f"{id_candidata}.parar-{n}"))
                    motivo = "parar"
                    break
                if not dir_salida.exists():
                    dir_salida.mkdir(parents=True)
                reporte = self.evaluar_candidata(dir_salida, id_candidata, propuesta.get("justificacion"),
                                                 propuesta.get("decidio", "llm"), t_decision,
                                                 mejor_anterior=estado["mejor"]["costo_total"],
                                                 episodio={"n": n, "paso": estado["pasos"]})
                politica.incorporar(recibir_reporte(reporte, id_candidata, dir_salida))
            if pausa:
                self.estado_corrida = "en_pausa: esperando acción del agente"
                break
            entradas_ep = [e for e in self.memoria.entradas if (e.get("episodio") or {}).get("n") == n]
            mejor = estado_episodio(entradas_ep, referencia, tolerancia)["mejor"]
            mejora = mejor["costo_total"] < c_best - tolerancia
            atribuibles = [e for e in entradas_ep if e["estado_evaluacion"] not in estados.NO_ATRIBUIBLES]
            episodios.append({
                "n": n, "semilla": self.corrida.get("semilla", 0) + n, "pasos": len(entradas_ep),
                "mejor": mejor["id_evaluacion"], "malla": mejor.get("malla"), "C": mejor["costo_total"],
                "estado": estados.CORRECTA if atribuibles or not entradas_ep else estados.ERROR_INFRAESTRUCTURA,
                "mejora": mejora, "C_best": min(c_best, mejor["costo_total"]), "fin": motivo,
                "decisiones": [e.get("decidio") for e in entradas_ep],
            })
            self._escribir_episodios(episodios)
        self._escribir_episodios(self.episodios(), self.estado_corrida)
        self.escribir_resumen(self.estado_corrida)
        from ..herramientas.convergencia import escribir as escribir_convergencia
        escribir_convergencia(self.carpeta)
        if self.corrida.get("informes", True):
            from ..herramientas.resultados import escribir_resultados
            escribir_resultados(self.carpeta)
        return self.estado_corrida

    # --- registro ---------------------------------------------------------
    def escribir_resumen(self, estado_corrida=None):
        entradas = self.memoria.entradas
        agente = [e for e in entradas if not e.get("es_referencia")]
        estado = self.memoria.reconstruir()
        correctos = [e["costo_total"] for e in entradas if e["estado_evaluacion"] == estados.CORRECTA]
        validas = [e for e in agente if e["estado_evaluacion"] != estados.ACCION_INVALIDA]
        resumen = {
            "id": self.corrida["id"],
            "estado": estado_corrida or self.estado_corrida or "en_curso",
            "epsilon": self.corrida.get("epsilon", 0.0),
            "semilla": self.corrida.get("semilla", 0),
            "actualizado": ahora(),
            "C_min": estado["C_min"],
            "optimas": [o["id_candidata"] for o in estado["optimas"]],
            "evaluaciones": len(entradas),
            "evaluaciones_agente": len(agente),
            "iteraciones_sin_mejora": estado["sin_mejora"],
            "conteo_por_estado": self.memoria.conteo(),
            "porcentaje_validas": round(100 * len(validas) / len(agente), 1) if agente else None,
            "porcentaje_correctas": round(100 * sum(e["estado_evaluacion"] == estados.CORRECTA for e in agente)
                                          / len(agente), 1) if agente else None,
            "evolucion_costo": [{"id": e["id_evaluacion"], "estado": e["estado_evaluacion"],
                                 "costo_total": e["costo_total"], "recompensa": e["recompensa"]} for e in entradas],
            "tiempo_evaluacion_total_s": round(sum(e.get("tiempo_evaluacion_s") or 0 for e in entradas), 3),
            "normalizadores": self.normalizadores,
            "avisos_penalizacion": verificar_penalizacion(self.corrida["penalizacion"]["P_fallo"], correctos),
            "nota": "Óptimas encontradas dentro del presupuesto; no garantiza el óptimo global.",
        }
        escribir_json(self.carpeta / "resumen.json", resumen)
        return resumen
