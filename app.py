import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from datetime import timedelta

# --- Import project modules (assumes this file is at repo root) ---
try:
    from src.data_preprocessing import basic_preprocessing
    from src.feature_engineering import feature_engineering
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    from src.data_preprocessing import basic_preprocessing
    from src.feature_engineering import feature_engineering

MODEL_PATH = os.getenv("MODEL_PATH", "models/Current_model.pkl")
PIPE_1 = os.getenv("PIPE_1", "pipelines/preprocessing_pipeline.pkl")
PIPE_2 = os.getenv("PIPE_2", "pipelines/featureSelection_pipeline.pkl")
DEFAULT_CSV = os.getenv("DEFAULT_CSV", "data/latest_3_year.csv")

_model, _pipe1, _pipe2 = None, None, None

def load_artifacts():
    global _model, _pipe1, _pipe2
    if _pipe1 is None:
        _pipe1 = joblib.load(PIPE_1)
    if _pipe2 is None:
        _pipe2 = joblib.load(PIPE_2)
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model, _pipe1, _pipe2

def _postprocess_prediction(dates, y_pred):
    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "horizon_day": np.arange(1, len(y_pred) + 1),
        "pred_temp (°C)": np.round(np.asarray(y_pred, dtype=float), 2),
    })
    return out

def _make_plot(df_out: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=120)
    ax.plot(df_out["date"], df_out[df_out.columns[-1]], marker="o", color="#0ea5e9", linewidth=2)
    ax.fill_between(df_out["date"], df_out[df_out.columns[-1]], color="#0ea5e922")
    ax.set_title("🌡️ 5-day Temperature Forecast", fontsize=12, fontweight="bold", color="#0369a1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def predict_from_csv(csv_path):
    try:
        model, pipe1, pipe2 = load_artifacts()
    except Exception as e:
        return None, gr.update(value=f"❌ Artifact error: {e}"), None

    if not csv_path or (isinstance(csv_path, float) and pd.isna(csv_path)):
        csv_path = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None
    try:
        if not csv_path:
            raise ValueError("No CSV provided and DEFAULT_CSV not found.")
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        return None, gr.update(value=f"❌ CSV error: {e}"), None

    try:
        df_prep = basic_preprocessing(df_raw)
        if not isinstance(df_prep.index, pd.DatetimeIndex):
            raise ValueError("Datetime index not set after preprocessing.")

        last_dt = df_prep.index.max()
        df_processed = pipe1.transform(df_prep)
        df_feat, target_cols = feature_engineering(df_processed)
        X = df_feat.drop(columns=target_cols)
        X_sel = pipe2.transform(X)
        x_last = X_sel.tail(1)
        if x_last.shape[0] == 0:
            raise ValueError("No valid rows after feature engineering.")

        y_pred = model.predict(x_last)[0]
        horizon_dates = [(last_dt + timedelta(days=i+1)).date() for i in range(len(y_pred))]
        df_out = _postprocess_prediction(horizon_dates, y_pred)

        summary = "\n".join([f"Day +{r.horizon_day} ({r.date.date()}): {r._3} °C" for r in df_out.itertuples()])
        fig = _make_plot(df_out)
        return df_out, summary, fig

    except Exception as e:
        return None, gr.update(value=f"❌ Pipeline error: {e}"), None

with gr.Blocks(
    title="Hanoi Temperature Forecast – Gradio UI",
    theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"),
    css="""
    body {background: radial-gradient(900px 400px at 10% -10%, #dbeafe66, transparent),
                   radial-gradient(900px 400px at 90% 0%, #c7d2fe66, transparent);}
    .gradio-container {max-width: 1180px}
    .hero-title {font-size: 30px; font-weight: 800; letter-spacing: 0.2px;}
    .cool-card {background: linear-gradient(135deg, #0ea5e922, #64748b22); border: 1px solid #0ea5e933; border-radius: 14px; padding: 14px;}
    .cold-btn {border-radius: 12px !important; background-color:#0284c7 !important; color:white !important;}
    .result-card {background: #ffffffcc; border: 1px solid #e2e8f0; border-radius: 14px;}
    """
) as demo:
    gr.Markdown(
        """
        <div class=\"hero-title\">🌤️ Hanoi Temperature Forecast — 5 days</div>
        <div class=\"cool-card\">Dự báo nhanh 5 ngày kế tiếp. Có thể upload CSV hoặc để app tự chạy với dữ liệu mặc định trong máy chủ.</div>
        """
    )

    with gr.Row():
        csv_in = gr.File(label="Upload daily CSV (raw)", file_types=[".csv"], type="filepath")

    with gr.Row():
        btn = gr.Button("🚀 Dự báo 5 ngày", elem_classes=["cold-btn"])

    with gr.Row():
        df_out = gr.Dataframe(label="Kết quả dự báo (°C)", interactive=False)
    with gr.Row():
        plot_out = gr.Plot(label="Biểu đồ dự báo (5 ngày)")

    txt = gr.Textbox(label="Tóm tắt", lines=6, elem_classes=["result-card"])

    demo.load(fn=lambda: predict_from_csv(DEFAULT_CSV), inputs=None, outputs=[df_out, txt, plot_out])
    btn.click(predict_from_csv, inputs=[csv_in], outputs=[df_out, txt, plot_out])

if __name__ == "__main__":
    demo.launch(share = True)