"""Contrato compartido con la solución RL (``solucion_rl.md``, sección 10).

Dijkstra + Greedy solo cambia *quién propone la configuración* (Greedy) y
*quién genera los PE*.txt* (Dijkstra). Todo lo demás debe ser idéntico a
``rl/``, y está aquí, con las mismas reglas y los mismos nombres:

* estados de evaluación (``rl/estados.py``);
* validación previa de la acción (``rl/agente/paquete.py``);
* validación funcional y C_R (``rl/medicion/funcional.py``);
* área, latencia y costo normalizado (``rl/medicion/metricas.py``);
* agregación entre operaciones (``rl/medicion/conjunto.py``);
* recompensa (``rl/medicion/recompensa.py``) y óptimas (``rl/agente/optimas.py``).

Los parámetros viven en ``costos.json``, con los valores de la corrida RL.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1] / "config"
RUTA_COSTOS = CARPETA / "costos.json"

# -- estados (rl/estados.py) ------------------------------------------------------
ACCION_INVALIDA = "accion_invalida"
FALLO_EVALUADOR = "fallo_evaluador"
FALLO_SIMULADOR = "fallo_simulador"
INCORRECTA = "incorrecta"
CORRECTA = "correcta"
ERROR_INFRAESTRUCTURA = "error_infraestructura"
ESTADOS = (ACCION_INVALIDA, FALLO_EVALUADOR, FALLO_SIMULADOR, INCORRECTA, CORRECTA, ERROR_INFRAESTRUCTURA)
PENALIZADOS = frozenset({ACCION_INVALIDA, FALLO_EVALUADOR, FALLO_SIMULADOR, INCORRECTA})
NO_ATRIBUIBLES = frozenset({ERROR_INFRAESTRUCTURA})
# Si varias operaciones fallan, la configuración toma el primero de esta lista que aparezca.
PRIORIDAD_FALLA = (ERROR_INFRAESTRUCTURA, ACCION_INVALIDA, FALLO_EVALUADOR, FALLO_SIMULADOR, INCORRECTA)

FORMULAS_LATENCIA = ("ciclos", "ciclos_x_nops")
FORMULA_RECOMPENSA = "r = C_best_prev - C (correcta); r = -P_fallo (penalizada); None (error_infraestructura)"
FORMULA_COSTO = ("C = promedio sobre las operaciones de (K_A*C_A/C_A_ref + K_L*C_L,op/C_L_ref,op + K_R*C_R,op)")
MEMORIA_OPS = frozenset({"LD", "ST", "VLD", "VST"})
COMPUTO_OPS = frozenset({"ADD", "SUB", "MUL", "DIV", "VADD", "VSUB", "VMUL", "VDIV"})


class ErrorReferencia(RuntimeError):
    """La referencia no es correcta: sin ella no hay normalización (el RL se detiene)."""


# -- parámetros (costos.json) -------------------------------------------------------
@dataclass
class Costos:
    """Parámetros de evaluación, los mismos de ``corrida.json`` en el RL."""

    pesos_area: dict
    capacidades: dict  # "MS" -> frozenset de instrucciones
    K: dict
    formula_latencia: str = "ciclos_x_nops"
    tolerancias: dict = field(default_factory=lambda: {"eps_abs": 1e-6, "eps_rel": 1e-6, "epsilon_cr": 1e-9})
    p_fallo: float = 10.0
    tolerancia_empate: float = 1e-9
    t_max_simulacion_s: float = 60
    restricciones: dict = field(default_factory=dict)
    referencia: list = field(default_factory=lambda: ["M E M", "M V M", "M E M"])
    ruta_normalizadores: Path = None

    PESO_DE = {"MS": "w1_MS", "MV": "w2_MV", "CS": "w3_CS", "CV": "w4_CV"}

    @classmethod
    def cargar(cls, ruta=RUTA_COSTOS):
        ruta = Path(ruta)
        d = json.loads(ruta.read_text(encoding="utf-8"))
        if d["formula_latencia"] not in FORMULAS_LATENCIA:
            raise ValueError(f"{ruta}: formula_latencia debe ser una de {FORMULAS_LATENCIA}")
        capacidades = {c: frozenset(i.upper() for i in lista) for c, lista in d["capacidades"].items()}
        faltan = [cls.PESO_DE[c] for c in capacidades if cls.PESO_DE.get(c) not in d["pesos_area"]]
        if faltan:
            raise ValueError(f"{ruta}: faltan pesos de área: {', '.join(faltan)}")
        normalizadores = d["referencia"].get("normalizadores")
        return cls(
            pesos_area=dict(d["pesos_area"]), capacidades=capacidades, K=dict(d["K"]),
            formula_latencia=d["formula_latencia"], tolerancias=dict(d["tolerancias"]),
            p_fallo=d["penalizacion"]["P_fallo"], tolerancia_empate=d["tolerancia_empate_costo"],
            t_max_simulacion_s=d["limites"]["t_max_simulacion_s"], restricciones=dict(d["restricciones"]),
            referencia=list(d["referencia"]["malla"]),
            ruta_normalizadores=(ruta.parent / normalizadores) if normalizadores else None,
        )

    # Área (rl/medicion/metricas.py): cada PE suma las capacidades que declara.
    def capacidades_pe(self, instrucciones):
        instrucciones = {i.upper() for i in instrucciones}
        return [c for c, lista in self.capacidades.items() if lista & instrucciones]

    def area_pe(self, instrucciones):
        return sum(self.pesos_area[self.PESO_DE[c]] for c in self.capacidades_pe(instrucciones))

    def conteo_capacidades(self, formato):
        conteo = dict.fromkeys(self.capacidades, 0)
        for fila in formato.malla:
            for letra in fila:
                for c in self.capacidades_pe(formato.tipos_pe[letra]["instrucciones"]):
                    conteo[c] += 1
        return conteo

    def area(self, formato):
        return sum(self.pesos_area[self.PESO_DE[c]] * n for c, n in self.conteo_capacidades(formato).items())

    def latencia(self, ciclos, nops, n_pes):
        """C_L de una operación: ``ciclos`` o ``ciclos·(1 + NOPs/(ciclos·PEs))``."""
        if ciclos is None:
            return None
        if self.formula_latencia == "ciclos":
            return ciclos
        fraccion = fraccion_nops(ciclos, nops, n_pes)
        if fraccion is None:
            raise ValueError("'ciclos_x_nops' necesita los NOPs y el número de PEs")
        return ciclos * (1 + fraccion)

    def criterio(self, normalizadores):
        """Lo que debe ser igual en todas las evaluaciones de una búsqueda."""
        return dict(pesos_area=self.pesos_area, K=self.K, formula_latencia=self.formula_latencia,
                    tolerancias=self.tolerancias, penalizacion={"P_fallo": self.p_fallo},
                    tolerancia_empate_costo=self.tolerancia_empate,
                    capacidades={c: sorted(i) for c, i in self.capacidades.items()},
                    normalizadores=normalizadores)

    def parametros_usados(self, normalizadores):
        """``parametros`` del reporte E08 del RL (rl/medicion/reporte.py)."""
        return dict(pesos_area=self.pesos_area, K=self.K, formula_latencia=self.formula_latencia,
                    tolerancias=self.tolerancias, penalizacion={"P_fallo": self.p_fallo},
                    tolerancia_empate_costo=self.tolerancia_empate, normalizadores=normalizadores,
                    formula_recompensa=FORMULA_RECOMPENSA, formula_costo=FORMULA_COSTO)


def fraccion_nops(ciclos, nops, n_pes):
    """NOPs / (ciclos·PEs): proporción de ranuras de instrucción en NOP."""
    if ciclos is None or nops is None or not n_pes or ciclos <= 0:
        return None
    return nops / (ciclos * n_pes)


# -- normalizadores ------------------------------------------------------------------
def cargar_normalizadores(costos):
    """Normalizadores congelados (``normalizacion.json``), o ``None`` si no hay archivo."""
    if costos.ruta_normalizadores is None:
        return None
    datos = json.loads(Path(costos.ruta_normalizadores).read_text(encoding="utf-8"))
    formula = datos.get("formula_latencia")
    if formula is not None and formula != costos.formula_latencia:
        raise ValueError(f"{costos.ruta_normalizadores} se midió con formula_latencia = '{formula}' y "
                         f"costos.json usa '{costos.formula_latencia}'")
    return {"C_A_ref": datos["C_A_ref"],
            "operaciones": {op: {"C_A_ref": d.get("C_A_ref", datos["C_A_ref"]), "C_L_ref": d["C_L_ref"]}
                            for op, d in datos["operaciones"].items()},
            "referencia": datos.get("referencia"), "fuente": datos.get("fuente")}


# -- validación previa (rl/agente/paquete.py) ------------------------------------------
def validar_accion(datos, formato, restricciones, tipos_esquema):
    """Motivos por los que la configuración es ``accion_invalida`` (lista vacía si es válida).

    ``datos``: el dict del cgra.json; ``formato``: su ``FormatoCGRA`` ya cargado;
    ``tipos_esquema``: ``tipos_pe`` del cgra.json real."""
    motivos = []
    r = restricciones
    for eje, valor in (("filas", formato.filas), ("columnas", formato.columnas)):
        minimo, maximo = r[eje].get("minimo", 1), r[eje].get("maximo")
        if valor < minimo or (maximo is not None and valor > maximo):
            motivos.append(f"{eje} = {valor} fuera del rango permitido [{minimo}, {maximo}]")
    if r.get("palabras_memoria_max") is not None and formato.palabras_memoria > r["palabras_memoria_max"]:
        motivos.append(f"palabras_memoria = {formato.palabras_memoria} supera {r['palabras_memoria_max']}")
    if formato.ancho_vector is not None and r.get("anchos_vector_permitidos") and \
            formato.ancho_vector not in r["anchos_vector_permitidos"]:
        motivos.append(f"ancho_vector = {formato.ancho_vector} no está en {r['anchos_vector_permitidos']}")
    if "tipos_pe" not in datos:
        motivos.append("cgra.json debe declarar 'tipos_pe' (tomados del esquema real)")
    usados = {letra for fila in formato.malla for letra in fila}
    for letra, tipo in formato.tipos_pe.items():
        if letra not in tipos_esquema:
            motivos.append(f"tipo de PE '{letra}' no existe en el esquema de la corrida")
        elif {i.upper() for i in tipo["instrucciones"]} != {i.upper() for i in tipos_esquema[letra]["instrucciones"]}:
            motivos.append(f"el tipo '{letra}' cambia su lista de instrucciones respecto al esquema real")
    if r.get("requiere_memoria_y_computo"):
        ops = set().union(*({i.upper() for i in formato.tipos_pe[l]["instrucciones"]} for l in usados))
        if not ops & MEMORIA_OPS:
            motivos.append("la malla no tiene ningún PE de memoria")
        if not ops & COMPUTO_OPS:
            motivos.append("la malla no tiene ningún PE de cómputo")
    if r.get("memoria_solo_en_bordes_oeste_este"):
        motivos += _motivos_bordes(formato)
    return motivos


def _motivos_bordes(formato):
    """Memoria en los bordes: oeste y este son M; el interior, cómputo sin acceso a memoria."""
    if formato.columnas < 3:
        return [f"con {formato.columnas} columna(s) no hay interior para el cómputo: la memoria va en las "
                "columnas oeste y este y el cómputo entre ellas (se necesitan al menos 3 columnas)"]
    motivos = []
    bordes = (0, formato.columnas - 1)
    for f, fila in enumerate(formato.malla):
        for c, letra in enumerate(fila):
            accede_memoria = bool({i.upper() for i in formato.tipos_pe[letra]["instrucciones"]} & MEMORIA_OPS)
            if c in bordes and letra != "M":
                motivos.append(f"PE({f},{c}) es {letra}: los bordes oeste y este deben ser de memoria (M)")
            elif c not in bordes and accede_memoria:
                motivos.append(f"PE({f},{c}) es {letra} y accede a memoria: la memoria solo puede ir en los "
                               "bordes oeste/este; el interior es de cómputo (E, V o C)")
    return motivos


# -- validación funcional (rl/medicion/funcional.py) -----------------------------------
def comparar(esperado, obtenido, tolerancias, max_fallos=10):
    """``esperado``/``obtenido``: ``{región: [valores]}``. Correcta solo si **todos** los elementos
    cumplen |y − ŷ| ≤ eps_abs + eps_rel·|y|; C_R = promedio de |y − ŷ| / (|y| + ε)."""
    eps_abs, eps_rel, epsilon_cr = tolerancias["eps_abs"], tolerancias["eps_rel"], tolerancias["epsilon_cr"]
    total, fallidos, suma_error = 0, 0, 0.0
    primeros, estructurales = [], []
    for nombre, valores in esperado.items():
        if nombre not in obtenido:
            estructurales.append(f"falta la región '{nombre}' en la memoria final")
            continue
        if len(obtenido[nombre]) != len(valores):
            estructurales.append(f"'{nombre}': {len(obtenido[nombre])} valores y se esperaban {len(valores)}")
            continue
        for i, (y, y_hat) in enumerate(zip(valores, obtenido[nombre])):
            total += 1
            if not math.isfinite(y_hat):
                fallidos += 1
                suma_error = math.inf
                if len(primeros) < max_fallos:
                    primeros.append({"region": nombre, "i": i, "esperado": y, "obtenido": str(y_hat)})
                continue
            diferencia = abs(y - y_hat)
            suma_error += diferencia / (abs(y) + epsilon_cr)
            if diferencia > eps_abs + eps_rel * abs(y):
                fallidos += 1
                if len(primeros) < max_fallos:
                    primeros.append({"region": nombre, "i": i, "esperado": y, "obtenido": y_hat})
    aprobada = not estructurales and fallidos == 0 and total > 0
    c_r = None if estructurales or total == 0 else suma_error / total
    if c_r is not None and not math.isfinite(c_r):
        c_r = None
    return {"aprobada": aprobada, "total_elementos": total, "elementos_fallidos": fallidos,
            "primeros_fallos": primeros, "errores_estructurales": estructurales, "C_R": c_r}


# -- recompensa y óptimas (rl/medicion/recompensa.py, rl/agente/optimas.py) --------------
def costo_reportado(estado, costo_medido, p_fallo):
    if estado == CORRECTA:
        return costo_medido
    if estado in PENALIZADOS:
        return p_fallo
    return None


def recompensa(estado, costo, mejor_costo_anterior, p_fallo):
    """``r = C_best_prev − C`` si es correcta, ``−P_fallo`` si se penaliza, ``None`` si es infraestructura."""
    if estado in NO_ATRIBUIBLES:
        return None
    if estado in PENALIZADOS:
        return -p_fallo
    if mejor_costo_anterior is None:
        raise ErrorReferencia("no hay C_best previo: la referencia debe evaluarse correctamente antes de optimizar")
    return mejor_costo_anterior - costo


def actualizar_optimas(optimas, entrada, tolerancia):
    """Devuelve ``(nuevas_optimas, C_min, hubo_mejora)``; solo participan las ``correcta``."""
    actuales = list(optimas)
    c_min = min((o["costo_total"] for o in actuales), default=None)
    if entrada["estado_evaluacion"] != CORRECTA or entrada.get("costo_total") is None:
        return actuales, c_min, False
    costo = entrada["costo_total"]
    candidata = {"id_candidata": entrada["accion"]["id_candidata"], "hash_accion": entrada["accion"]["hash_accion"],
                 "directorio": entrada["accion"].get("directorio"), "costo_total": costo,
                 "metricas": entrada["metricas"], "es_referencia": entrada.get("es_referencia", False),
                 "malla": entrada.get("malla")}
    if c_min is None or costo < c_min - tolerancia:
        return [candidata], costo, True
    if abs(costo - c_min) <= tolerancia and all(o["hash_accion"] != candidata["hash_accion"] for o in actuales):
        actuales.append(candidata)
    return actuales, c_min, False


def verificar_penalizacion(p_fallo, costos_correctos):
    maximo = max(costos_correctos, default=None)
    if maximo is not None and p_fallo <= maximo:
        return [f"P_fallo = {p_fallo} no supera el mayor costo correcto observado ({maximo})"]
    return []


def hash_configuracion(configuracion):
    texto = json.dumps(configuracion, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()
