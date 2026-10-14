"""App Streamlit — Árbol de decisión aplicado a accidentes aéreos.
Ejecutar:  streamlit run app.py
"""
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import plot_tree
import modelo as M

st.set_page_config(page_title="Árbol de decisión · Accidentes aéreos", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Entrenando el árbol de decisión…")
def cargar():
    r = M.entrenar()
    r["cv"] = M.validacion_cruzada(r["modelo"], r["X"], r["y"])
    return r


R = cargar()
modelo, df, X = R["modelo"], R["df"], R["X"]
mt = R["metricas"]["Árbol ajustado (prueba)"]
mb = R["metricas"]["Línea base"]
cv = R["cv"]


def bonito(c):
    for pref, nom in [("operador_cat_", "Operador: "), ("familia_avion_", "Avión: "),
                      ("region_", "Región: "), ("fase_vuelo_", "Fase: ")]:
        if c.startswith(pref):
            return nom + c[len(pref):]
    return M.ETIQUETAS.get(c, c)


NOMBRES = [bonito(c) for c in X.columns]
CLASES = ["Con sobrevivientes", "Pérdida total"]

st.title("✈️ Árbol de decisión: ¿un accidente aéreo deja sobrevivientes?")
st.caption(f"Dataset: {len(df):,} accidentes (1950–2009) · Modelo: DecisionTreeClassifier (scikit-learn)")

t1, t2, t3, t4, t5, t6 = st.tabs(["📘 Guía", "📊 Datos", "🎯 Evaluación", "🌳 El árbol", "🧪 Simulador", "💼 Conclusiones"])

# ---------------------------------------------------------------- GUÍA
with t1:
    st.markdown(f"""
### ¿Qué hace esta aplicación?
Entrena un **árbol de decisión** que predice si un accidente aéreo será de **pérdida total**
(nadie sobrevive) o si **habrá sobrevivientes**, a partir de las circunstancias del accidente.

### ¿Cómo se usa?
1. Revisa **Datos** para entender el problema y el balance de clases.
2. En **Evaluación** mira las métricas y su interpretación (matriz de confusión, ROC, umbral).
3. En **El árbol** ve las reglas que aprendió el modelo.
4. En **Simulador** ingresa un escenario y obtén la predicción, su probabilidad y el camino de reglas.

### ¿Cómo funciona un árbol de decisión?
Hace preguntas de sí/no sobre las variables (p. ej. *¿ocupantes ≤ 12?*), eligiendo en cada paso la que
mejor separa las clases (reduce la **impureza de Gini**). Termina en hojas con una probabilidad.

| Ventajas | Limitaciones | Supuestos / cuidados |
|---|---|---|
| Fácil de explicar y visualizar | Se **sobreajusta** si crece sin límite | Los datos de prueba deben ser independientes del entrenamiento |
| No requiere escalar variables | Inestable: pequeños cambios alteran el árbol | Se limitó la profundidad y se podó (`ccp_alpha`) |
| Captura relaciones no lineales | Fronteras "escalonadas", menos preciso que ensambles | Clases moderadamente desbalanceadas (65/35) |
""")

# ---------------------------------------------------------------- DATOS
with t2:
    p = df.perdida_total.mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Accidentes analizados", f"{len(df):,}")
    c2.metric("Pérdida total (clase 1)", f"{p:.1%}")
    c3.metric("Con sobrevivientes (clase 0)", f"{1-p:.1%}")
    st.info(f"**Clave para las métricas:** si el modelo dijera siempre «pérdida total» acertaría {p:.1%} de las veces. "
            "Ese es el piso que debe superar: por eso la exactitud sola engaña.")
    a, b = st.columns(2)
    d = df.assign(rango=pd.cut(df.ocupantes, [0, 3, 10, 30, 100, 1000], labels=["1–3", "4–10", "11–30", "31–100", ">100"]))
    g = d.groupby("rango", observed=True).perdida_total.mean().reset_index()
    a.plotly_chart(px.bar(g, x="rango", y="perdida_total", title="% de pérdida total según ocupantes",
                          labels={"perdida_total": "% pérdida total", "rango": "Ocupantes"}).update_yaxes(tickformat=".0%"), width="stretch")
    g = df.groupby("fase_vuelo").perdida_total.mean().reset_index()
    b.plotly_chart(px.bar(g, x="fase_vuelo", y="perdida_total", title="% de pérdida total según fase de vuelo",
                          labels={"perdida_total": "% pérdida total", "fase_vuelo": "Fase"}).update_yaxes(tickformat=".0%"), width="stretch")
    a, b = st.columns(2)
    g = df.groupby(["operador_cat"]).perdida_total.agg(["mean", "size"]).reset_index()
    a.plotly_chart(px.bar(g, x="operador_cat", y="mean", title="% de pérdida total según operador",
                          labels={"mean": "% pérdida total", "operador_cat": ""}).update_yaxes(tickformat=".0%"), width="stretch")
    g = df.groupby((df.anio // 10) * 10).size().reset_index(name="n")
    b.plotly_chart(px.bar(g, x="anio", y="n", title="Accidentes por década", labels={"anio": "Década", "n": "Accidentes"}), width="stretch")
    with st.expander("Ver datos preparados"):
        st.dataframe(df[["anio", "mes", "hora", "ocupantes", "operador_cat", "familia_avion", "region",
                         "fase_vuelo", "clima_adverso", "terreno_montanoso", "perdida_total"]].head(200))

# ---------------------------------------------------------------- EVALUACIÓN
with t3:
    st.subheader("Métricas en el conjunto de prueba (20 % de los datos, nunca vistos)")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, mt.items()):
        col.metric(k, f"{v:.3f}", f"{v - mb[k]:+.3f} vs. línea base")
    tabla = pd.DataFrame({**R["metricas"]}).T.round(3)
    st.dataframe(tabla, width="stretch")
    st.markdown(f"""
**Cómo leerlo**
- **Exactitud ({mt['Exactitud (accuracy)']:.1%})** es casi igual a la línea base ({mb['Exactitud (accuracy)']:.1%}): sola **no demuestra** que el modelo sirva.
- **AUC ({mt['AUC-ROC']:.2f})**: probabilidad de que el modelo asigne mayor riesgo a un accidente de pérdida total que a uno con sobrevivientes
  (0.5 = azar, 1 = perfecto). Es la métrica más justa aquí porque no depende del umbral ni del desbalance. La línea base tiene 0.50.
- **Precisión ({mt['Precisión']:.1%})**: cuando el modelo dice «pérdida total», acierta esa proporción de veces.
- **Recall ({mt['Recall (sensibilidad)']:.1%})**: de los accidentes de pérdida total reales, detecta esa proporción.
- **F1** equilibra precisión y recall. Ojo: la línea base tiene F1 alto solo porque predice siempre la clase 1 (recall 100 %, AUC 0.5).
""")
    a, b = st.columns(2)
    m = R["matriz"]
    fig = px.imshow(m, text_auto=True, x=CLASES, y=CLASES, color_continuous_scale="Blues",
                    labels=dict(x="Predicho", y="Real", color="Casos"), title="Matriz de confusión (prueba)")
    a.plotly_chart(fig, width="stretch")
    tn, fp, fn, tp = m.ravel()
    a.markdown(f"- **{tp}** pérdidas totales bien detectadas (VP) y **{tn}** accidentes con sobrevivientes bien identificados (VN).\n"
               f"- **{fp}** falsas alarmas (se predijo pérdida total y sí hubo sobrevivientes).\n"
               f"- **{fn}** pérdidas totales no detectadas (FN): el error más costoso si la decisión es de rescate.")
    fpr, tpr = R["roc"]; fl, tl = R["roc_libre"]
    f = go.Figure()
    f.add_scatter(x=fpr, y=tpr, name=f"Árbol ajustado (AUC {mt['AUC-ROC']:.2f})")
    f.add_scatter(x=fl, y=tl, name=f"Árbol sin límites (AUC {R['metricas']['Árbol sin límites (prueba)']['AUC-ROC']:.2f})", line=dict(dash="dot"))
    f.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color="gray"))
    f.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Recall")
    b.plotly_chart(f, width="stretch")

    st.subheader("Sobreajuste: el árbol sin límites memoriza")
    a, b = st.columns(2)
    cc = R["curva_complejidad"].melt("profundidad", var_name="Conjunto", value_name="Exactitud")
    a.plotly_chart(px.line(cc, x="profundidad", y="Exactitud", color="Conjunto", markers=True,
                           title="Exactitud vs. profundidad del árbol"), width="stretch")
    ml = R["metricas_train"]["Árbol sin límites"]["Exactitud (accuracy)"]
    b.markdown(f"""
El árbol sin límites logra **{ml:.0%}** en entrenamiento pero solo **{R['metricas']['Árbol sin límites (prueba)']['Exactitud (accuracy)']:.0%}** en prueba: memorizó el ruido.
Con `GridSearchCV` (validación cruzada, criterio AUC) se eligió: **{R['mejores_params']}**.
Al aumentar la profundidad, la exactitud de entrenamiento sube pero la de prueba se estanca o baja.
""")

    st.subheader("Validación cruzada (5×5): ¿qué tan estable es el resultado?")
    st.dataframe(cv.rename(columns={"media": "Media", "desv": "Desv. estándar"}).round(3), width="stretch")
    st.caption(f"Un solo split da AUC {mt['AUC-ROC']:.2f}; la validación cruzada da {cv.loc['AUC-ROC','media']:.2f} ± {cv.loc['AUC-ROC','desv']:.2f}. "
               "Se reporta la variación porque un único split puede ser optimista o pesimista.")

    st.subheader("Umbral de decisión: precisión vs. recall")
    cu = M.curva_umbral(R["y_te_arr"], R["prob_te"])
    st.plotly_chart(px.line(cu.melt("umbral", var_name="Métrica", value_name="Valor"), x="umbral", y="Valor", color="Métrica"),
                    width="stretch")
    st.caption("Umbral bajo → se detectan más pérdidas totales (recall alto) pero con más falsas alarmas. "
               "El umbral correcto depende de qué error cuesta más.")

# ---------------------------------------------------------------- ÁRBOL
with t4:
    prof = st.slider("Niveles a mostrar", 1, int(modelo.get_depth()), min(3, int(modelo.get_depth())))
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(modelo, feature_names=NOMBRES, class_names=CLASES, filled=True, rounded=True,
              max_depth=prof, fontsize=9, ax=ax, impurity=False, proportion=True)
    st.pyplot(fig)
    st.caption("Cada nodo muestra la regla, el % de casos que llegan a él y la clase mayoritaria (color: azul = pérdida total, naranja = con sobrevivientes).")
    imp = R["importancias"].head(8).rename(index=bonito).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Importancia de variables (reducción de impureza)",
                           labels={"value": "Importancia", "index": ""}).update_layout(showlegend=False), width="stretch")

# ---------------------------------------------------------------- SIMULADOR
with t5:
    st.subheader("Simulador de escenarios")
    a, b, c = st.columns(3)
    ocup = a.number_input("Ocupantes a bordo", 1, 700, 8)
    anio = a.slider("Año", 1950, 2009, 1995)
    mes = a.slider("Mes", 1, 12, 6)
    hora = a.slider("Hora (0–23)", 0, 23, 14)
    oper = b.selectbox("Operador", sorted(df.operador_cat.unique()))
    fam = b.selectbox("Familia de avión", sorted(df.familia_avion.unique()))
    reg = b.selectbox("Región", sorted(df.region.unique()))
    fase = c.selectbox("Fase de vuelo", sorted(df.fase_vuelo.unique()), index=0)
    clima = c.checkbox("Clima adverso (niebla, tormenta, hielo…)")
    mont = c.checkbox("Terreno montañoso")
    agua = c.checkbox("Sobre agua")
    inc = c.checkbox("Incendio o falla de motor")
    umbral = st.slider("Umbral de decisión", 0.05, 0.95, 0.50, 0.05)

    ent = dict(anio=anio, mes=mes, hora=hora, ocupantes=ocup, clima_adverso=int(clima), terreno_montanoso=int(mont),
               sobre_agua=int(agua), incendio_o_motor=int(inc), operador_cat=oper, familia_avion=fam, region=reg, fase_vuelo=fase)
    fila = M.fila_desde_entrada(ent, X.columns)
    prob = float(modelo.predict_proba(fila)[0, 1])
    veredicto = "PÉRDIDA TOTAL" if prob >= umbral else "CON SOBREVIVIENTES"
    r1, r2 = st.columns([1, 1])
    g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": " %"},
                               title={"text": "Probabilidad de pérdida total"},
                               gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 4}, "value": umbral * 100},
                                      "bar": {"color": "#1f4e79"}}))
    r1.plotly_chart(g, width="stretch")
    (r2.error if prob >= umbral else r2.success)(f"### Predicción: {veredicto}")
    r2.write(f"El modelo estima **{prob:.0%}** de probabilidad de que no haya sobrevivientes (umbral {umbral:.0%}).")
    t = modelo.tree_
    nodos = modelo.decision_path(fila).indices
    r2.markdown("**Camino de reglas que siguió el árbol:**")
    for n in nodos[:-1]:
        f = t.feature[n]; v = fila.iloc[0, f]; th = t.threshold[n]
        r2.markdown(f"- {NOMBRES[f]} = **{v:g}** → {'≤' if v <= th else '>'} {th:.1f}")
    hoja = nodos[-1]
    cnt = t.value[hoja][0]; cnt = cnt / cnt.sum()
    r2.caption(f"Hoja final: {int(t.n_node_samples[hoja])} accidentes de entrenamiento con perfil parecido (proporciones ponderadas por clase: {cnt[1]:.0%} pérdida total).")
    st.warning("Herramienta académica: explora patrones históricos de 1950–2009; no sirve para evaluar la seguridad de un vuelo real.")

# ---------------------------------------------------------------- CONCLUSIONES
with t6:
    top = R["importancias"].head(3).rename(index=bonito)
    st.markdown(f"""
### Resultado
El árbol logra **AUC ≈ {mt['AUC-ROC']:.2f}** en prueba (≈ {cv.loc['AUC-ROC','media']:.2f} en validación cruzada) frente a 0.50 de la línea base,
con precisión de **{mt['Precisión']:.0%}** y recall de **{mt['Recall (sensibilidad)']:.0%}**. Tiene **capacidad moderada**: ordena el riesgo mejor que el azar, pero no es un predictor fino.
Las variables que más pesan: **{', '.join(top.index)}**.

### ¿Qué decisión tomaría una empresa o autoridad?
- **Seguridad en aviación ligera y regional:** los accidentes con pocos ocupantes y durante despegue/aproximación son los de mayor pérdida total.
  Priorizar balizas de localización (ELT), tiempos de respuesta de rescate y entrenamiento en esas fases.
- **Aseguradoras:** usar la probabilidad como señal de severidad para tarifar, no como dictamen individual.
- **Elegir el umbral según el costo del error:** si dejar de detectar una pérdida total es lo más grave, bajar el umbral y aceptar más falsas alarmas.

### Limitaciones honestas
- El dataset solo contiene accidentes con víctimas (1950–2009): no mide la probabilidad de accidente, solo el desenlace dado que ocurrió.
- Fase de vuelo y condiciones se extrajeron del texto del resumen con reglas simples; hay ruido.
- Son asociaciones históricas, no causas. Un modelo más potente (Random Forest) podría mejorar el AUC a costa de interpretabilidad.
""")
