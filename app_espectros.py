import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ======================================================
# Configuración de la App
# ======================================================
st.set_page_config(
    page_title="Analizador UV-Vis Profesional",
    layout="wide"
)

st.title("📊 Sistema Avanzado de Espectros y Linealidad UV-Vis")

LAMBDA_MIN_ABS = 190
LAMBDA_MAX_ABS = 800

# ======================================================
# Funciones Auxiliares
# ======================================================
@st.cache_data
def leer_csv(file):
    file.seek(0)
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower()
    return df

def extraer_factor_dilucion(texto):
    numeros = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
    if not numeros:
        return 1.0
    return float(numeros[-1]) if float(numeros[-1]) > 0 else 1.0

# ======================================================
# Panel de Control - Barra Lateral
# ======================================================
st.sidebar.header("1. Cargar archivos")
archivos = st.sidebar.file_uploader(
    "Selecciona tus archivos CSV", type="csv", accept_multiple_files=True
)

etiquetas = {}

if archivos:
    st.sidebar.header("2. Identificar diluciones")
    st.sidebar.markdown("_Ejemplos: 1:1, 1:2, 1:5, 1:10, Pura_")

    for archivo in archivos:
        etiquetas[archivo.name] = st.sidebar.text_input(
            f"Dilución para {archivo.name}", value="1:1", key=archivo.name
        )

    st.sidebar.header("3. Opciones de visualización")
    aplicar_correccion = st.sidebar.checkbox(
        "🔄 Aplicar absorbancia corregida en espectros (A × Factor)", value=False
    )

    st.sidebar.header("4. Región de picos")
    rango_picos = st.sidebar.slider(
        "Rango (nm) para buscar picos máximos:",
        min_value=LAMBDA_MIN_ABS, max_value=LAMBDA_MAX_ABS, value=(268, 288), step=1
    )
    pico_min, pico_max = rango_picos

    generar = st.sidebar.button("Procesar Datos", type="primary")

    if "procesado" not in st.session_state:
        st.session_state.procesado = False

    if generar:
        st.session_state.procesado = True

    # ======================================================
    # Procesamiento de Datos y Generación de Espectros
    # ======================================================
    if st.session_state.procesado:
        
        # Crear pestañas para organizar el flujo de trabajo
        tab1, tab2 = st.tabs(["📈 Barridos de Absorbancia", "🧪 Estudio de Linealidad (Beer-Lambert)"])
        
        fig_espectros = go.Figure()
        datos_graficados = 0
        lista_picos_calibracion = []

        for archivo in archivos:
            try:
                df = leer_csv(archivo)
                if "nm" not in df.columns or "a" not in df.columns:
                    continue

                df["nm"] = pd.to_numeric(df["nm"], errors="coerce")
                df["a"] = pd.to_numeric(df["a"], errors="coerce")
                df = df.dropna(subset=["nm", "a"])

                df_grafico = df[(df["nm"] >= LAMBDA_MIN_ABS) & (df["nm"] <= LAMBDA_MAX_ABS)].copy()
                if df_grafico.empty:
                    continue

                texto_dilucion = etiquetas[archivo.name]
                factor = extraer_factor_dilucion(texto_dilucion)
                
                # Concentración absoluta calculada como 1 / factor de dilución
                concentracion_absoluta = 1.0 / factor
                
                factor_espectro = factor if aplicar_correccion else 1.0
                df_grafico["y_render"] = df_grafico["a"] * factor_espectro

                nombre = Path(archivo.name).stem
                leyenda = f"{nombre} [{texto_dilucion}]"

                # Curva del espectro
                fig_espectros.add_trace(go.Scatter(
                    x=df_grafico["nm"], y=df_grafico["y_render"],
                    mode="lines", name=leyenda, showlegend=True
                ))

                # Buscar pico en la región original medida
                df_region = df_grafico[(df_grafico["nm"] >= pico_min) & (df_grafico["nm"] <= pico_max)]
                if not df_region.empty:
                    idx_max = df_region["a"].idxmax()
                    pico_x = df_region.loc[idx_max, "nm"]
                    pico_y_puro = df_region.loc[idx_max, "a"]
                    
                    # Añadir marca visual en la pestaña 1
                    pico_y_render = df_region.loc[idx_max, "y_render"]
                    fig_espectros.add_trace(go.Scatter(
                        x=[pico_x], y=[pico_y_render], mode="markers",
                        marker=dict(size=10, symbol="x"), showlegend=False
                    ))

                    # Guardar estructura para el análisis de linealidad
                    lista_picos_calibracion.append({
                        "id": archivo.name,
                        "Muestra": nombre,
                        "Dilución": texto_dilucion,
                        "Factor": factor,
                        "Concentración (1/Factor)": concentracion_absoluta,
                        "λ Pico (nm)": round(pico_x, 2),
                        "Absorbancia Original (A)": pico_y_puro
                    })
                datos_graficados += 1
            except Exception as e:
                st.error(f"Error en {archivo.name}: {e}")

        # --- PESTAÑA 1: VISTA DE ESPECTROS ---
        with tab1:
            st.subheader("Visualización de Barridos Espectrales")
            fig_espectros.update_layout(
                template="plotly_white", hovermode="x unified",
                xaxis=dict(title="Longitud de onda (nm)", range=[LAMBDA_MIN_ABS, LAMBDA_MAX_ABS]),
                yaxis=dict(title="Absorbancia Corregida" if aplicar_correccion else "Absorbancia Pura (A)"),
                height=550
            )
            # Solución al error de sintaxis previo (falta de asignación '=')
            st.plotly_chart(fig_espectros, use_container_width=True)
            
            # Tabla de picos resumida
            df_resumen_picos = pd.DataFrame(lista_picos_calibracion)
            if not df_resumen_picos.empty:
                st.dataframe(df_resumen_picos.drop(columns=["id", "Factor"]), use_container_width=True, hide_index=True)

        # --- PESTAÑA 2: ESTUDIO DE LINEALIDAD ---
        with tab2:
            st.subheader("Análisis de Linealidad y Rango Dinámico (Beer-Lambert)")
            
            if len(lista_picos_calibracion) < 2:
                st.info("Necesitas cargar al menos 2 archivos con diluciones distintas para calcular la linealidad.")
            else:
                df_calib = pd.DataFrame(lista_picos_calibracion)
                
                st.markdown("### 🛠️ Optimización del Coeficiente de Determinación ($R^2$)")
                st.markdown("Desmarca las muestras que se salgan de la linealidad (por ejemplo, soluciones saturadas o demasiado diluidas) para recalcular el ajuste.")
                
                # Crear columnas: izquierda para seleccionar datos, derecha para ver gráfico de calibración
                col_filtros, col_grafico = st.columns([1, 2])
                
                with col_filtros:
                    st.write("**Muestras a incluir:**")
                    puntos_seleccionados = []
                    
                    # Generar un checkbox para cada muestra detectada
                    for idx, row in df_calib.iterrows():
                        incluir = st.checkbox(
                            f"{row['Muestra']} ({row['Dilución']})", 
                            value=True, 
                            key=f"check_{row['id']}"
                        )
                        if incluir:
                            puntos_seleccionados.append(row["id"])
                
                # Filtrar el DataFrame según la selección del usuario
                df_filtrado_calib = df_calib[df_calib["id"].isin(puntos_seleccionados)].copy()
                
                with col_grafico:
                    if len(df_filtrado_calib) < 2:
                        st.error("⚠️ Debes mantener al menos 2 puntos seleccionados para trazar una recta.")
                    else:
                        # Extraer variables X e Y para la regresión lineal
                        X = df_filtrado_calib[["Concentración (1/Factor)"]].values
                        Y = df_filtrado_calib["Absorbancia Original (A)"].values
                        
                        # Ajustar modelo lineal
                        modelo = LinearRegression()
                        modelo.fit(X, Y)
                        Y_pred = modelo.predict(X)
                        
                        # Calcular el R2
                        r2 = r2_score(Y, Y_pred)
                        pendiente = modelo.coef_[0]
                        intercepto = modelo.intercept_
                        
                        # Graficar los puntos y la recta de ajuste
                        fig_linealidad = go.Figure()
                        
                        # Puntos reales medidos
                        fig_linealidad.add_trace(go.Scatter(
                            x=df_filtrado_calib["Concentración (1/Factor)"],
                            y=df_filtrado_calib["Absorbancia Original (A)"],
                            mode="markers",
                            marker=dict(size=12, color="blue"),
                            name="Datos Experimentales",
                            text=df_filtrado_calib["Dilución"],
                            hovertemplate="<b>Dilución: %{text}</b><br>Conc: %{x:.4f}<br>Abs: %{y:.4f}<extra></extra>"
                        ))
                        
                        # Recta de ajuste ideal calculada
                        x_linea = np.linspace(X.min(), X.max(), 100)
                        y_linea = pendiente * x_linea + intercepto
                        fig_linealidad.add_trace(go.Scatter(
                            x=x_linea, y=y_linea,
                            mode="lines",
                            line=dict(dash="dash", color="red"),
                            name="Ajuste Lineal"
                        ))
                        
                        # --- CONFIGURACIÓN ÓPTIMA SIN DISTORSIÓN ---
                        fig_linealidad.update_layout(
                            title="Curva de Calibración: Concentración vs Absorbancia",
                            template="plotly_white",
                            margin=dict(l=60, r=40, t=50, b=50),
                            width=550,
                            height=550,
                            xaxis=dict(
                                title="Concentración Absoluta (1 / Factor de Dilución)",
                                autorange=True
                            ),
                            yaxis=dict(
                                title="Absorbancia Pura Máxima (A)",
                                autorange=True
                            )
                        )
                        st.plotly_chart(fig_linealidad, use_container_width=False)
                        
                        # --- VEREDICTO DE LINEALIDAD ---
                        st.markdown(f"#### El coeficiente de determinación actual es: **$R^2 = {r2:.5f}$**")
                        
                        if r2 >= 0.97:
                            st.success(f"🎉 **¡Linealidad Respetada!** El nivel de $R^2$ ({r2:.4f}) cumple con el criterio mínimo de 0.97.")
                            diluciones_validas = df_filtrado_calib["Dilución"].tolist()
                            st.write(f"Los factores de dilución donde hay linealidad son: **{', '.join(diluciones_validas)}**")
                        else:
                            st.error(f"❌ **El $R^2$ es muy bajo, no hay linealidad.** ($R^2 = {r2:.4f} < 0.97$).")
                            st.markdown(
                                "> **Consejo de laboratorio:** Intenta desmarcar en la columna izquierda los puntos extremos. "
                                "Las concentraciones altas suelen desviar la curva hacia abajo por saturación química."
                            )
else:
    st.info("👈 Carga uno o más archivos CSV desde la barra lateral y presiona 'Procesar Datos'.")