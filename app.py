import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RPM Order & Vibration Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = ["Time (s)", "ChA (m/s²)", "ChB (m/s²)", "ChC (m/s²)", "RPM"]
CHANNELS = ["ChA (m/s²)", "ChB (m/s²)", "ChC (m/s²)"]

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .metric-card {
        background: linear-gradient(135deg, rgba(38,99,235,.12), rgba(14,165,233,.08));
        border: 1px solid rgba(148,163,184,.35);
        padding: 18px; border-radius: 18px;
    }
    .small-muted {color: #64748b; font-size: 0.92rem;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_excel(uploaded_bytes: bytes, sheet_name: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name=sheet_name, engine="openpyxl")


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def vibration_summary(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    rows = []
    for ch in channels:
        data = safe_numeric(df[ch]).dropna()
        if len(data) == 0:
            continue
        rows.append({
            "Channel": ch,
            "Samples": int(len(data)),
            "RMS (m/s²)": float(np.sqrt(np.mean(np.square(data)))),
            "Peak (m/s²)": float(np.max(np.abs(data))),
            "Peak-to-Peak (m/s²)": float(data.max() - data.min()),
            "Mean (m/s²)": float(data.mean()),
            "Std Dev (m/s²)": float(data.std()),
        })
    return pd.DataFrame(rows)


def rpm_summary(df: pd.DataFrame) -> pd.DataFrame:
    rpm = safe_numeric(df["RPM"]).dropna()
    return pd.DataFrame([{
        "RPM Min": float(rpm.min()),
        "RPM Max": float(rpm.max()),
        "RPM Mean": float(rpm.mean()),
        "RPM Std Dev": float(rpm.std()),
        "Samples": int(len(rpm)),
    }])


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, math.ceil(len(df) / max_points))
    return df.iloc[::step, :].copy()


def create_result_excel(df: pd.DataFrame, vib_df: pd.DataFrame, rpm_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Converted Data", index=False)
        vib_df.to_excel(writer, sheet_name="Vibration Summary", index=False)
        rpm_df.to_excel(writer, sheet_name="RPM Summary", index=False)
    output.seek(0)
    return output.getvalue()


def plot_time_series(df_plot: pd.DataFrame, selected_channels: list[str]):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = safe_numeric(df_plot["Time (s)"])
    for ch in selected_channels:
        ax.plot(x, safe_numeric(df_plot[ch]), label=ch, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Vibration Time Signal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_rpm(df_plot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(safe_numeric(df_plot["Time (s)"]), safe_numeric(df_plot["RPM"]), linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.set_title("RPM vs Time")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_order_cut(df_plot: pd.DataFrame, selected_channel: str, order_value: float):
    # Basit görsel order yaklaşımı: zaman alanı sinyal büyüklüğü ile RPM ilişkisi.
    # Gelişmiş order tracking için encoder/tach phase data gerekir.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    rpm = safe_numeric(df_plot["RPM"])
    y = safe_numeric(df_plot[selected_channel]).abs()
    ax.scatter(rpm, y, s=8, alpha=0.45)
    ax.set_xlabel("RPM")
    ax.set_ylabel(f"|{selected_channel}| (m/s²)")
    ax.set_title(f"Order View Placeholder: {order_value:g}x Order / {selected_channel}")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


st.title("RPM Order & Vibration Analyzer")
st.caption("Excel yükle, titreşim metriklerini hesapla, RPM trendini incele ve sonucu Excel olarak indir.")

with st.sidebar:
    st.header("Ayarlar")
    uploaded_file = st.file_uploader("ConvertedData Excel dosyasını yükle", type=["xlsx"])
    max_points = st.slider("Grafik maksimum nokta sayısı", 1_000, 100_000, 20_000, step=1_000)
    order_value = st.number_input("Order değeri", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
    st.divider()
    st.markdown("**Beklenen kolonlar**")
    st.code("\n".join(REQUIRED_COLUMNS), language="text")

if uploaded_file is None:
    st.info("Başlamak için sol menüden Excel dosyanı yükle.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()

with st.spinner("Excel okunuyor ve analiz hazırlanıyor..."):
    df = load_excel(uploaded_bytes)

missing = validate_columns(df)
if missing:
    st.error("Excel içinde beklenen kolonlar bulunamadı: " + ", ".join(missing))
    st.write("Bulunan kolonlar:", list(df.columns))
    st.stop()

for col in REQUIRED_COLUMNS:
    df[col] = safe_numeric(df[col])

available_channels = [c for c in CHANNELS if c in df.columns]
selected_channels = st.sidebar.multiselect("Grafikte gösterilecek kanallar", available_channels, default=available_channels)
selected_order_channel = st.sidebar.selectbox("Order/RPM görünümü kanalı", available_channels)

vib_df = vibration_summary(df, available_channels)
rpm_df = rpm_summary(df)
df_plot = downsample(df[REQUIRED_COLUMNS], max_points)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Toplam Satır", f"{len(df):,}")
with c2:
    st.metric("RPM Ortalama", f"{rpm_df.loc[0, 'RPM Mean']:.1f}")
with c3:
    st.metric("RPM Min-Max", f"{rpm_df.loc[0, 'RPM Min']:.0f} - {rpm_df.loc[0, 'RPM Max']:.0f}")
with c4:
    duration = df["Time (s)"].max() - df["Time (s)"].min()
    st.metric("Süre", f"{duration:.2f} s")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Özet", "Grafikler", "Order / RPM", "Data & İndir"])

with tab1:
    st.subheader("Vibration Summary")
    st.dataframe(vib_df, use_container_width=True, hide_index=True)
    st.subheader("RPM Summary")
    st.dataframe(rpm_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Time Signal")
    if selected_channels:
        plot_time_series(df_plot, selected_channels)
    else:
        st.warning("En az bir kanal seçmelisin.")
    st.subheader("RPM Trend")
    plot_rpm(df_plot)

with tab3:
    st.subheader(f"{order_value:g}x Order / RPM Görünümü")
    st.caption("Not: Bu ekran RPM ile titreşim genliği ilişkisini gösterir. Gerçek order tracking için tach/encoder faz bilgisi gerekir.")
    plot_order_cut(df_plot, selected_order_channel, order_value)

with tab4:
    st.subheader("Veri Önizleme")
    st.dataframe(df.head(1000), use_container_width=True)
    excel_bytes = create_result_excel(df, vib_df, rpm_df)
    st.download_button(
        "Analiz Sonucunu Excel Olarak İndir",
        data=excel_bytes,
        file_name="Vibration_Analysis_Result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
