"""
modelo.py — Preparación de datos y entrenamiento del Árbol de Decisión
=======================================================================
Problema: dado el contexto de un accidente aéreo (año, hora, ocupantes,
tipo de operador, familia de avión, fase de vuelo, condiciones...),
predecir si el accidente fue de PÉRDIDA TOTAL (nadie sobrevivió) o si
hubo sobrevivientes.

Variable objetivo (y):
    1 = ningún sobreviviente  (fatalities == aboard)
    0 = al menos un sobreviviente

Este módulo lo usan tanto la app (app.py) como cualquier notebook.
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42
RUTA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "plane_crashes_data.csv")

# ----------------------------------------------------------------------
# 1. LIMPIEZA E INGENIERÍA DE VARIABLES
# ----------------------------------------------------------------------
def _hora_a_numero(valor):
    """La columna 'hour' trae texto sucio ('c:09:00', "12'20", '?'). Extraemos la hora."""
    m = re.search(r"(\d{1,2})", str(valor))
    if m:
        h = int(m.group(1))
        return h if 0 <= h <= 23 else np.nan
    return np.nan


def _familia_avion(tipo):
    """Agrupa cientos de modelos en pocas familias (si no, el árbol memoriza)."""
    t = str(tipo).lower()
    reglas = [
        ("Douglas/McDonnell Douglas", ["douglas", "dc-", "c-47", "c-54"]),
        ("Boeing", ["boeing", "b-29", "b-17", "b-52"]),
        ("Lockheed", ["lockheed", "c-130", "l-188", "constellation"]),
        ("Cessna/Piper/Beech (ligeros)", ["cessna", "piper", "beech", "learjet", "mitsubishi", "swearingen", "britten", "partenavia"]),
        ("de Havilland", ["havilland"]),
        ("Antonov/Tupolev/Yakovlev/Ilyushin (URSS)", ["antonov", "tupolev", "yakovlev", "ilyushin", "an-", "tu-", "yak", "il-"]),
        ("Fokker/Convair/Vickers/Viscount", ["fokker", "convair", "vickers", "viscount", "curtiss", "martin"]),
        ("Airbus/ATR/CASA/Embraer", ["airbus", "atr", "casa", "embraer", "saab", "fairchild", "bae", "british aerospace", "hawker", "shorts"]),
    ]
    for nombre, claves in reglas:
        if any(c in t for c in claves):
            return nombre
    return "Otros"


def _categoria_operador(op):
    o = str(op).lower()
    if o.startswith("military"):
        return "Militar"
    if any(k in o for k in ["private", "taxi", "charter", "air tax"]):
        return "Privado / Taxi aéreo"
    return "Aerolínea comercial"


def _region(loc):
    """Región a partir del último término de 'location' (país o estado de EE. UU.)."""
    ultimo = str(loc).split(",")[-1].strip().lower()
    estados_us = {"alaska", "california", "new york", "colorado", "arizona", "hawaii",
                  "washington", "texas", "pennsylvania", "florida", "illinois", "ohio",
                  "georgia", "louisiana", "mississippi", "minnesota", "virginia",
                  "massachusetts", "michigan", "new jersey", "nevada", "oregon",
                  "utah", "montana", "idaho", "wyoming", "maryland", "missouri",
                  "north carolina", "south carolina", "tennessee", "kentucky",
                  "indiana", "wisconsin", "oklahoma", "new mexico", "kansas",
                  "nebraska", "iowa", "arkansas", "alabama", "connecticut",
                  "maine", "west virginia", "north dakota", "south dakota",
                  "puerto rico", "delaware", "vermont", "new hampshire", "rhode island",
                  "washington d.c.", "district of columbia", "usa", "united states"}
    if ultimo in estados_us:
        return "Estados Unidos"
    if ultimo in {"canada", "mexico", "guatemala", "honduras", "panama", "cuba", "haiti",
                  "dominican republic", "costa rica", "nicaragua", "el salvador", "jamaica"}:
        return "Canadá/México/Caribe/C. América"
    if ultimo in {"colombia", "brazil", "venezuela", "peru", "bolivia", "ecuador", "argentina",
                  "chile", "paraguay", "uruguay", "guyana", "suriname"}:
        return "Sudamérica"
    if ultimo in {"russia", "ukraine", "kazakhstan", "uzbekistan", "georgia", "armenia",
                  "azerbaijan", "belarus", "lithuania", "latvia", "estonia", "moldova",
                  "kyrgyzstan", "tajikistan", "turkmenistan", "siberia"}:
        return "Rusia y ex-URSS"
    if ultimo in {"italy", "france", "england", "spain", "germany", "switzerland", "austria",
                  "scotland", "ireland", "wales", "netherlands", "belgium", "norway", "sweden",
                  "denmark", "finland", "greece", "portugal", "poland", "czechoslovakia",
                  "yugoslavia", "romania", "bulgaria", "hungary", "iceland", "turkey",
                  "united kingdom", "croatia", "serbia", "bosnia"}:
        return "Europa"
    if ultimo in {"india", "indonesia", "philippines", "japan", "china", "taiwan", "thailand",
                  "vietnam", "pakistan", "nepal", "iran", "iraq", "south korea", "korea",
                  "malaysia", "burma", "myanmar", "afghanistan", "bangladesh", "sri lanka",
                  "laos", "cambodia", "papua new guinea", "israel", "lebanon", "saudi arabia",
                  "syria", "india.", "mongolia", "oman", "yemen", "uae", "kuwait"}:
        return "Asia/Oceanía/Medio Oriente"
    if "ocean" in ultimo or "sea" in ultimo or "gulf" in ultimo:
        return "Océano / Mar"
    return "África y otros"


def preparar_datos(ruta=RUTA_CSV):
    """Lee el CSV, limpia y construye la matriz de variables (X) y el objetivo (y)."""
    df = pd.read_csv(ruta)

    # Registros inválidos: 0 ocupantes no permite definir sobrevivientes
    df = df[df["aboard"] > 0].copy()

    # Objetivo: 1 = pérdida total (nadie sobrevivió)
    df["perdida_total"] = (df["fatalities"] >= df["aboard"]).astype(int)

    # Hora: texto sucio -> número; los faltantes se imputan con la mediana
    df["hora"] = df["hour"].apply(_hora_a_numero)
    df["hora"] = df["hora"].fillna(df["hora"].median())

    df["mes"] = df["month"]
    df["anio"] = df["year"]
    df["ocupantes"] = df["aboard"]

    df["operador_cat"] = df["operator"].apply(_categoria_operador)
    df["familia_avion"] = df["type"].apply(_familia_avion)
    df["region"] = df["location"].apply(_region)

    # Variables derivadas del resumen (solo circunstancias, NUNCA el desenlace)
    s = df["summary"].fillna("").str.lower()
    df["fase_vuelo"] = np.select(
        [s.str.contains("take-?off|takeoff|after departure|shortly after leaving"),
         s.str.contains("approach|landing|landed|touchdown|runway"),
         s.str.contains("en route|cruise|cruising|climb|descen")],
        ["Despegue", "Aproximación/Aterrizaje", "En ruta"],
        default="No especificada")
    df["clima_adverso"] = s.str.contains("fog|storm|weather|thunder|rain|snow|icing|ice |wind|visibility|turbulence|cloud").astype(int)
    df["terreno_montanoso"] = s.str.contains("mountain|mount |hill|peak|ridge|slope|terrain").astype(int)
    df["sobre_agua"] = s.str.contains("ocean|sea |water|lake|river|bay|gulf").astype(int)
    df["incendio_o_motor"] = s.str.contains("fire|engine|explo").astype(int)

    return df


NUMERICAS = ["anio", "mes", "hora", "ocupantes", "clima_adverso",
             "terreno_montanoso", "sobre_agua", "incendio_o_motor"]
CATEGORICAS = ["operador_cat", "familia_avion", "region", "fase_vuelo"]

ETIQUETAS = {
    "anio": "Año", "mes": "Mes", "hora": "Hora del día", "ocupantes": "Ocupantes a bordo",
    "clima_adverso": "Clima adverso", "terreno_montanoso": "Terreno montañoso",
    "sobre_agua": "Sobre agua", "incendio_o_motor": "Incendio / falla de motor",
}


def construir_X(df, columnas_dummies=None):
    """One-hot encoding de las categóricas (los árboles de sklearn requieren números)."""
    X = pd.get_dummies(df[NUMERICAS + CATEGORICAS], columns=CATEGORICAS, dtype=int)
    if columnas_dummies is not None:
        X = X.reindex(columns=columnas_dummies, fill_value=0)
    return X


# ----------------------------------------------------------------------
# 2. ENTRENAMIENTO Y EVALUACIÓN
# ----------------------------------------------------------------------
def _metricas(y_real, y_pred, y_prob):
    return {
        "Exactitud (accuracy)": accuracy_score(y_real, y_pred),
        "Precisión": precision_score(y_real, y_pred, zero_division=0),
        "Recall (sensibilidad)": recall_score(y_real, y_pred),
        "F1": f1_score(y_real, y_pred),
        "AUC-ROC": roc_auc_score(y_real, y_prob),
    }


def entrenar():
    """
    Devuelve un diccionario con el modelo final, métricas, curvas y datos de apoyo.
    Pasos:
      1) Split 80/20 estratificado (conserva la proporción de clases).
      2) Línea base: predecir siempre la clase mayoritaria.
      3) Árbol SIN límites (muestra el sobreajuste).
      4) Árbol AJUSTADO con GridSearchCV + validación cruzada de 5 pliegues.
    """
    df = preparar_datos()
    X = construir_X(df)
    y = df["perdida_total"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA)

    # --- Línea base ---
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    m_base = _metricas(y_te, base.predict(X_te), base.predict_proba(X_te)[:, 1])

    # --- Árbol sin podar (sobreajuste) ---
    libre = DecisionTreeClassifier(random_state=SEMILLA).fit(X_tr, y_tr)
    m_libre = _metricas(y_te, libre.predict(X_te), libre.predict_proba(X_te)[:, 1])
    m_libre_train = _metricas(y_tr, libre.predict(X_tr), libre.predict_proba(X_tr)[:, 1])

    # --- Árbol ajustado ---
    rejilla = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [5, 10, 20, 40],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.003],
        "class_weight": [None, "balanced"],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=SEMILLA), rejilla,
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), n_jobs=-1)
    gs.fit(X_tr, y_tr)
    modelo = gs.best_estimator_

    prob_te = modelo.predict_proba(X_te)[:, 1]
    pred_te = modelo.predict(X_te)
    m_final = _metricas(y_te, pred_te, prob_te)
    m_final_train = _metricas(y_tr, modelo.predict(X_tr), modelo.predict_proba(X_tr)[:, 1])
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    fpr_l, tpr_l, _ = roc_curve(y_te, libre.predict_proba(X_te)[:, 1])

    # --- Curva de complejidad: exactitud train vs test según profundidad ---
    curva = []
    for d in range(1, 16):
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=gs.best_params_["min_samples_leaf"],
                                   random_state=SEMILLA).fit(X_tr, y_tr)
        curva.append({"profundidad": d,
                      "Entrenamiento": t.score(X_tr, y_tr),
                      "Prueba": t.score(X_te, y_te)})
    curva = pd.DataFrame(curva)

    importancias = (pd.Series(modelo.feature_importances_, index=X.columns)
                    .sort_values(ascending=False))

    return {
        "df": df, "X": X, "y": y, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "modelo": modelo, "mejores_params": gs.best_params_, "auc_cv": gs.best_score_,
        "metricas": {"Línea base": m_base, "Árbol sin límites (prueba)": m_libre,
                     "Árbol ajustado (prueba)": m_final},
        "metricas_train": {"Árbol sin límites": m_libre_train, "Árbol ajustado": m_final_train},
        "matriz": confusion_matrix(y_te, pred_te),
        "roc": (fpr, tpr), "roc_libre": (fpr_l, tpr_l),
        "curva_complejidad": curva, "importancias": importancias,
        "prob_te": prob_te, "y_te_arr": y_te.values,
    }


def fila_desde_entrada(entrada, columnas):
    """Convierte lo que el usuario ingresa en la app en una fila alineada con el modelo."""
    fila = pd.DataFrame([entrada])
    X = pd.get_dummies(fila, columns=CATEGORICAS, dtype=int)
    return X.reindex(columns=columnas, fill_value=0)


if __name__ == "__main__":
    r = entrenar()
    print("Mejores hiperparámetros:", r["mejores_params"], "| AUC CV:", round(r["auc_cv"], 3))
    for k, v in r["metricas"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    for k, v in r["metricas_train"].items():
        print("TRAIN", k, {a: round(b, 3) for a, b in v.items()})
    print(r["matriz"])
    print(r["importancias"].head(10).round(3))
    print(r["df"].perdida_total.mean(), r["df"].region.value_counts().to_dict())
    print(r["df"].familia_avion.value_counts().to_dict())
    print(r["df"].fase_vuelo.value_counts().to_dict())


# ----------------------------------------------------------------------
# 3. APOYOS PARA INTERPRETAR (validación cruzada y umbral)
# ----------------------------------------------------------------------
def validacion_cruzada(modelo, X, y):
    """5 pliegues x 5 repeticiones: media y desviación de cada métrica (más estable que un solo split)."""
    from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEMILLA)
    r = cross_validate(modelo, X, y, cv=cv,
                       scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    nombres = {"accuracy": "Exactitud", "precision": "Precisión", "recall": "Recall",
               "f1": "F1", "roc_auc": "AUC-ROC"}
    return pd.DataFrame({n: [r["test_" + k].mean(), r["test_" + k].std()]
                         for k, n in nombres.items()}, index=["media", "desv"]).T


def curva_umbral(y, prob):
    """Precisión, recall y F1 al mover el umbral de decisión."""
    filas = []
    for u in np.linspace(0.05, 0.95, 91):
        p = (prob >= u).astype(int)
        filas.append({"umbral": u,
                      "Precisión": precision_score(y, p, zero_division=0),
                      "Recall": recall_score(y, p, zero_division=0),
                      "F1": f1_score(y, p, zero_division=0)})
    return pd.DataFrame(filas)
