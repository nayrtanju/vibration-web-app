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
ACC_FS = 26500
RPM_FS = 1024

G_TO_MS2 = 9.80665

ORDERS_TO_TRACK = [10, 20]

RPM_STEP = 10
ORDER_RESOLUTION = 0.125
ORDER_WIDTH = 0.15

MIN_BLOCK_SIZE = 1024
MAX_BLOCK_SIZE = 8192

RAW_SHEET_NAME = "Raw Data"

TIME_COL = "time"
VIBRATION_COLS = ["ChA", "ChB", "ChC"]
RPM_COL = "RPM"


# =====================================================
# READ EXCEL
# =====================================================
def read_excel_file(uploaded_file):
    try:
        return pd.read_excel(uploaded_file, sheet_name=RAW_SHEET_NAME)
    except Exception:
        return pd.read_excel(uploaded_file)


# =====================================================
# FIXED COLUMN STRUCTURE
# time, ChA, ChB, ChC, RPM
# =====================================================
def fix_column_structure(df):
    df = df.copy()

    expected_columns = [TIME_COL, "ChA", "ChB", "ChC", RPM_COL]

    if len(df.columns) < 5:
        raise ValueError(
            "Excel dosyasında en az 5 kolon olmalı: time, ChA, ChB, ChC, RPM"
        )

    df = df.iloc[:, :5]
    df.columns = expected_columns

    return df


# =====================================================
# UNIT CONVERSION
# =====================================================
def convert_units(raw_df):
    converted = raw_df.copy()

    converted[TIME_COL] = pd.to_numeric(converted[TIME_COL], errors="coerce")
    converted[RPM_COL] = pd.to_numeric(converted[RPM_COL], errors="coerce")

    for col in VIBRATION_COLS:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
        converted[f"{col}_m_s2"] = converted[col] * G_TO_MS2

    converted["Rotational_Frequency_Hz"] = converted[RPM_COL] / 60

    converted = converted.replace([np.inf, -np.inf], np.nan)

    converted = converted.dropna(
        subset=[TIME_COL, RPM_COL] + VIBRATION_COLS
    ).reset_index(drop=True)

    converted = converted[converted[RPM_COL] > 0].reset_index(drop=True)

    return converted


# =====================================================
# CALCULATE FFT BLOCK SIZE SAFELY
# =====================================================
def calculate_block_size(mean_rpm):
    shaft_freq = mean_rpm / 60

    if shaft_freq <= 0 or np.isnan(shaft_freq):
        return None

    freq_resolution = shaft_freq * ORDER_RESOLUTION

    if freq_resolution <= 0 or np.isnan(freq_resolution):
        return None

    block_size = int(ACC_FS / freq_resolution)
    block_size = int(2 ** np.ceil(np.log2(block_size)))

    block_size = max(MIN_BLOCK_SIZE, block_size)
    block_size = min(MAX_BLOCK_SIZE, block_size)

    return block_size


# =====================================================
# RPM BASED ORDER TRACKING
# =====================================================
def rpm_based_order_tracking(converted_df):
    if converted_df.empty:
        return pd.DataFrame()

    rpm_values = converted_df[RPM_COL].to_numpy()
    time_values = converted_df[TIME_COL].to_numpy()

    rpm_min = int(np.nanmin(rpm_values))
    rpm_max = int(np.nanmax(rpm_values))

    if rpm_min <= 0 or rpm_max <= 0 or rpm_max <= rpm_min:
        return pd.DataFrame()

    rpm_bins = np.arange(rpm_min, rpm_max + RPM_STEP, RPM_STEP)

    all_results = []

    for vib_col in VIBRATION_COLS:
        signal_col = f"{vib_col}_m_s2"

        if signal_col not in converted_df.columns:
            continue

        signal_values = converted_df[signal_col].to_numpy()

        for rpm_target in rpm_bins:
            mask = (
                (rpm_values >= rpm_target - RPM_STEP / 2) &
                (rpm_values < rpm_target + RPM_STEP / 2)
            )

            sample_count = np.sum(mask)

            if sample_count < MIN_BLOCK_SIZE:
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

            if np.allclose(signal_block, 0):
                continue

            window = get_window("hann", block_size)
            signal_windowed = signal_block * window

            spectrum = np.abs(rfft(signal_windowed)) * 2 / np.sum(window)
            freqs = rfftfreq(block_size, d=1 / ACC_FS)

            shaft_freq = mean_rpm / 60

            if shaft_freq <= 0:
                continue

            order_axis = freqs / shaft_freq

            row = {
                "Channel": vib_col,
                "RPM_Target": rpm_target,
                "RPM_Mean": mean_rpm,
                "Time_s": np.nanmean(time_block),
                "Sample_Count": sample_count,
                "Block_Size": block_size,
                "Order_Resolution": ORDER_RESOLUTION,
                "Order_Width": ORDER_WIDTH,
                "Window": "Hanning",
                "Tracking_Type": "RPM based"
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

    y_col = f"{order}th_Order_Amplitude_m_s2"

    channel_df = channel_df.dropna(subset=["RPM_Mean", y_col])

    if channel_df.empty:
        ax.set_title(f"{channel} - {order}th Order Cut: No valid data")
        ax.set_xlabel("RPM")
        ax.set_ylabel("Amplitude [m/s²]")
        ax.grid(True)
    else:
        ax.plot(
            channel_df["RPM_Mean"],
            channel_df[y_col],
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

    summary = pd.DataFrame({
        "Parameter": [
            "Column order",
            "Acceleration sampling rate",
            "RPM sampling frequency",
            "Orders",
            "RPM step",
            "Order resolution",
            "Order width",
            "Minimum block size",
            "Maximum block size",
            "Window",
            "Tracking type"
        ],
        "Value": [
            "time, ChA, ChB, ChC, RPM",
            ACC_FS,
            RPM_FS,
            str(ORDERS_TO_TRACK),
            RPM_STEP,
            ORDER_RESOLUTION,
            ORDER_WIDTH,
            MIN_BLOCK_SIZE,
            MAX_BLOCK_SIZE,
            "Hanning",
            "RPM based"
        ]
    })

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        converted_df.to_excel(writer, sheet_name="Converted Data", index=False)
        order_df.to_excel(writer, sheet_name="Order Cuts", index=False)
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
Sabit Excel kolon sırası:

1. time  
2. ChA  
3. ChB  
4. ChC  
5. RPM  

Analiz ayarları:
- Sampling rate: 26500 Hz
- RPM / CAN sampling frequency: 1024 Hz
- Sensor: Accelerometer
- Orders: 10th ve 20th
- Window: Hanning / Hann
- RPM step: 10 rpm
- Spectral resolution: 0.125 order
- Width: ±0.15 order
- Tracking: RPM based
""")

uploaded_file = st.file_uploader(
    "Excel dosyasını yükle",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        raw_df_original = read_excel_file(uploaded_file)
        raw_df = fix_column_structure(raw_df_original)

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head())

        converted_df = convert_units(raw_df)

        st.subheader("Converted Data Preview")
        st.dataframe(converted_df.head())

        st.info(f"Temizlenmiş data satır sayısı: {len(converted_df)}")

        with st.spinner("RPM based order tracking hesaplanıyor..."):
            order_df = rpm_based_order_tracking(converted_df)

        st.subheader("Order Cut Results")
        st.dataframe(order_df)

        png_buffers = {}

        for channel in VIBRATION_COLS:
            for order in ORDERS_TO_TRACK:
                st.subheader(f"{channel} - {order}th Order Cut")

                fig, png_buffer = create_order_plot(order_df, channel, order)
                st.pyplot(fig)

                file_name = f"{channel}_{order}th_order_cut.png"
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

        if order_df.empty:
            st.warning("""
            Order sonucu üretilemedi.

            Muhtemel sebepler:
            - RPM aralığında yeterli data yok
            - Her 10 RPM bin içinde minimum 1024 örnek yok
            - RPM datası sıfır/geçersiz
            - Data süresi çok kısa
            """)

    except Exception as e:
        st.error("Analiz sırasında hata oluştu.")
        st.exception(e)
