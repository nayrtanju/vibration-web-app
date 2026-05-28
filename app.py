import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
from io import BytesIO
import zipfile

# =====================================================
# SETTINGS
# =====================================================
ACC_FS = 26500          # ivme ölçer sampling rate
RPM_FS = 1024           # CAN / OBD RPM sampling frequency

G_TO_MS2 = 9.80665

ORDERS_TO_TRACK = [10, 20]

RPM_STEP = 10
ORDER_RESOLUTION = 0.125
ORDER_WIDTH = 0.15

WINDOW_TYPE = "hann"

RAW_SHEET_NAME = "Raw Data"


# =====================================================
# READ EXCEL
# =====================================================
def read_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=RAW_SHEET_NAME)
    except:
        df = pd.read_excel(uploaded_file)

    return df


# =====================================================
# DETECT COLUMNS FROM EXISTING EXCEL STRUCTURE
# =====================================================
def detect_columns(df):
    """
    Excel kolon isimleri değiştirilmez.

    Varsayım:
    1. kolon = zaman
    2. kolon = RPM
    3. kolon ve sonrası = ivme kanalları
    """

    time_col = df.columns[0]
    rpm_col = df.columns[1]
    vibration_cols = list(df.columns[2:])

    return time_col, rpm_col, vibration_cols


# =====================================================
# UNIT CONVERSION
# =====================================================
def convert_units(raw_df):
    converted = raw_df.copy()

    time_col, rpm_col, vibration_cols = detect_columns(converted)

    converted[time_col] = pd.to_numeric(converted[time_col], errors="coerce")
    converted[rpm_col] = pd.to_numeric(converted[rpm_col], errors="coerce")

    for col in vibration_cols:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
        converted[f"{col}_m_s2"] = converted[col] * G_TO_MS2

    converted["Rotational_Frequency_Hz"] = converted[rpm_col] / 60

    converted = converted.dropna(subset=[time_col, rpm_col]).reset_index(drop=True)

    return converted, time_col, rpm_col, vibration_cols


# =====================================================
# CALCULATE FFT BLOCK SIZE FROM ORDER RESOLUTION
# =====================================================
def calculate_block_size(mean_rpm):
    shaft_freq = mean_rpm / 60

    if shaft_freq <= 0:
        return None

    freq_resolution = shaft_freq * ORDER_RESOLUTION
    block_size = int(ACC_FS / freq_resolution)

    block_size = max(512, block_size)
    block_size = int(2 ** np.ceil(np.log2(block_size)))

    return block_size


# =====================================================
# RPM BASED ORDER TRACKING
# =====================================================
def rpm_based_order_tracking(converted_df, time_col, rpm_col, vibration_cols):
    rpm_values = converted_df[rpm_col].to_numpy()
    time_values = converted_df[time_col].to_numpy()

    rpm_min = int(np.nanmin(rpm_values))
    rpm_max = int(np.nanmax(rpm_values))

    rpm_bins = np.arange(rpm_min, rpm_max + RPM_STEP, RPM_STEP)

    all_results = []

    for vib_col in vibration_cols:
        signal_col = f"{vib_col}_m_s2"

        if signal_col not in converted_df.columns:
            continue

        signal_values = converted_df[signal_col].to_numpy()

        for rpm_target in rpm_bins:
            mask = (
                (rpm_values >= rpm_target - RPM_STEP / 2) &
                (rpm_values < rpm_target + RPM_STEP / 2)
            )

            if np.sum(mask) < 512:
                continue

            signal_block = signal_values[mask]
            rpm_block = rpm_values[mask]
            time_block = time_values[mask]

            mean_rpm = np.nanmean(rpm_block)

            block_size = calculate_block_size(mean_rpm)

            if block_size is None:
                continue

            if len(signal_block) < block_size:
                continue

            signal_block = signal_block[:block_size]
            signal_block = signal_block - np.nanmean(signal_block)

            window = get_window(WINDOW_TYPE, block_size)
            signal_windowed = signal_block * window

            spectrum = np.abs(rfft(signal_windowed)) * 2 / np.sum(window)
            freqs = rfftfreq(block_size, d=1 / ACC_FS)

            shaft_freq = mean_rpm / 60
            order_axis = freqs / shaft_freq

            row = {
                "Channel": vib_col,
                "RPM_Target": rpm_target,
                "RPM_Mean": mean_rpm,
                "Time_s": np.nanmean(time_block),
                "Block_Size": block_size,
                "Order_Resolution": ORDER_RESOLUTION,
                "Order_Width": ORDER_WIDTH,
                "Window": "Hanning"
            }

            for order in ORDERS_TO_TRACK:
                lower_order = order - ORDER_WIDTH
                upper_order = order + ORDER_WIDTH

                order_mask = (
                    (order_axis >= lower_order) &
                    (order_axis <= upper_order)
                )

                if np.any(order_mask):
                    local_spectrum = spectrum[order_mask]
                    local_orders = order_axis[order_mask]
                    local_freqs = freqs[order_mask]

                    max_index = np.argmax(local_spectrum)

                    amp_m_s2 = local_spectrum[max_index]
                    detected_order = local_orders[max_index]
                    detected_freq = local_freqs[max_index]
                else:
                    amp_m_s2 = np.nan
                    detected_order = np.nan
                    detected_freq = np.nan

                row[f"{order}th_Order_Amplitude_m_s2"] = amp_m_s2
                row[f"{order}th_Order_Amplitude_g"] = amp_m_s2 / G_TO_MS2
                row[f"{order}th_Detected_Order"] = detected_order
                row[f"{order}th_Detected_Frequency_Hz"] = detected_freq
                row[f"{order}th_Expected_Frequency_Hz"] = shaft_freq * order

            all_results.append(row)

    return pd.DataFrame(all_results)


# =====================================================
# PNG GRAPH
# =====================================================
def create_order_plot(order_df, channel, order):
    fig, ax = plt.subplots(figsize=(11, 5))

    channel_df = order_df[order_df["Channel"] == channel].copy()
    channel_df = channel_df.sort_values("RPM_Mean")

    ax.plot(
        channel_df["RPM_Mean"],
        channel_df[f"{order}th_Order_Amplitude_m_s2"],
        marker="o"
    )

    ax.set_title(f"{channel} - {order}th Order Cut vs RPM")
    ax.set_xlabel("RPM")
    ax.set_ylabel("Amplitude [m/s²]")
    ax.grid(True)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)

    return fig, buffer


# =====================================================
# EXCEL OUTPUT
# =====================================================
def create_excel_output(raw_df, converted_df, order_df):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        converted_df.to_excel(writer, sheet_name="Converted Data", index=False)
        order_df.to_excel(writer, sheet_name="Order Cuts", index=False)

        summary = pd.DataFrame({
            "Parameter": [
                "Acceleration sampling rate",
                "RPM sampling frequency",
                "Orders",
                "RPM step",
                "Order resolution",
                "Order width",
                "Window",
                "Tracking type"
            ],
            "Value": [
                ACC_FS,
                RPM_FS,
                str(ORDERS_TO_TRACK),
                RPM_STEP,
                ORDER_RESOLUTION,
                ORDER_WIDTH,
                "Hanning",
                "RPM based"
            ]
        })

        summary.to_excel(writer, sheet_name="Summary", index=False)

    buffer.seek(0)
    return buffer


# =====================================================
# ZIP OUTPUT
# =====================================================
def create_zip_output(excel_buffer, png_buffers):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("postprocess_result.xlsx", excel_buffer.getvalue())

        for filename, png_buffer in png_buffers.items():
            z.writestr(filename, png_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


# =====================================================
# STREAMLIT WEB UI
# =====================================================
st.set_page_config(
    page_title="RPM Based Order Tracking",
    layout="wide"
)

st.title("RPM Based Order Tracking Postprocess")

st.write("""
Bu uygulama Excel dosyasındaki mevcut kolon isimlerini değiştirmeden çalışır.

Varsayım:
- 1. kolon = zaman
- 2. kolon = RPM
- 3. kolon ve sonrası = ivme kanalları
""")

uploaded_file = st.file_uploader(
    "Excel dosyasını yükle",
    type=["xlsx"]
)

if uploaded_file is not None:

    raw_df = read_excel_file(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    converted_df, time_col, rpm_col, vibration_cols = convert_units(raw_df)

    st.subheader("Algılanan Kolonlar")
    st.write(f"Zaman kolonu: `{time_col}`")
    st.write(f"RPM kolonu: `{rpm_col}`")
    st.write(f"İvme kanalları: `{vibration_cols}`")

    st.subheader("Converted Data Preview")
    st.dataframe(converted_df.head())

    with st.spinner("RPM based order tracking hesaplanıyor..."):
        order_df = rpm_based_order_tracking(
            converted_df,
            time_col,
            rpm_col,
            vibration_cols
        )

    st.subheader("Order Cut Results")
    st.dataframe(order_df)

    png_buffers = {}

    if not order_df.empty:
        for channel in order_df["Channel"].unique():
            for order in ORDERS_TO_TRACK:

                st.subheader(f"{channel} - {order}th Order Cut")

                fig, png_buffer = create_order_plot(order_df, channel, order)
                st.pyplot(fig)

                file_name = f"{channel}_{order}th_order_cut.png"
                file_name = file_name.replace("/", "_").replace("\\", "_")

                png_buffers[file_name] = png_buffer

                st.download_button(
                    label=f"{channel} - {order}th Order PNG indir",
                    data=png_buffer,
                    file_name=file_name,
                    mime="image/png"
                )

        excel_buffer = create_excel_output(
            raw_df,
            converted_df,
            order_df
        )

        st.download_button(
            label="Excel sonucunu indir",
            data=excel_buffer,
            file_name="postprocess_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        zip_buffer = create_zip_output(excel_buffer, png_buffers)

        st.download_button(
            label="Tüm çıktıları ZIP olarak indir",
            data=zip_buffer,
            file_name="rpm_order_postprocess_outputs.zip",
            mime="application/zip"
        )

    else:
        st.warning("""
        Order sonucu üretilemedi.

        Muhtemel sebepler:
        - RPM aralığında yeterli data yok
        - RPM step içinde yeterli örnek yok
        - Block size datadan büyük kaldı
        - RPM datası sıfır veya geçersiz
        """)
