import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Deteksi Piring & Mangkok YoloV8",
    layout="centered"
)

# ================== HEADER ==================
st.markdown("""
<h1 style="text-align:center;">Deteksi Piring & Mangkok</h1>
<p style="text-align:center; color:gray;">
YOLOv8 Custom Dataset Piring dan Mangkok Object Detection
</p>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================== SIDEBAR ==================
st.sidebar.header("Pengaturan Deteksi")

conf = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25, 0.05,
    help="Semakin tinggi, semakin yakin deteksi yang ditampilkan"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Petunjuk:**
- Threshold rendah → deteksi lebih banyak
- Threshold tinggi → lebih akurat
- Video & webcam disarankan ≥ 0.4
""")

# ================== TABS ==================
tab_img, tab_vid, tab_cam = st.tabs(["Gambar", "Video", "Webcam"])

# ================== TAB IMAGE ==================
with tab_img:
    st.subheader("Deteksi Gambar")

    img_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if img_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_file.read())
            img_path = tmp.name

        image = Image.open(img_path).convert("RGB")
        results = model.predict(img_path, conf=conf, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input Image**")
            st.image(np.array(image), channels="RGB", use_container_width=True)
        with col2:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated_rgb, use_container_width=True)

        with st.expander("Detail Deteksi"):
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                score = float(box.conf[0])
                st.write(f"- **{label}** ({score:.2f})")

# ================== TAB VIDEO ==================
with tab_vid:
    st.subheader("Deteksi Video")

    vid_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if vid_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read())
            vid_path = tmp.name

        st.info("Video berhasil diupload. Klik tombol di bawah untuk memproses.")

        if st.button("Proses"):
            cap = cv2.VideoCapture(vid_path)

            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25

            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix="_hasil_deteksi.mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            progress = st.progress(0)
            i = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf, verbose=False)
                annotated = results[0].plot()
                out.write(annotated)

                i += 1
                progress.progress(min(i / total, 1.0))

            cap.release()
            out.release()
            progress.empty()

            st.success("Video selesai diproses.")

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Video Hasil Deteksi",
                    data=f,
                    file_name="hasil_deteksi_piring_mangkok.mp4",
                    mime="video/mp4"
                )

# ================== TAB WEBCAM ==================
with tab_cam:
    st.subheader("Deteksi Real-Time via Webcam")
    st.info("Izinkan akses kamera pada browser untuk memulai.")

    class YOLOVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.conf = conf

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, conf=self.conf, verbose=False)
            return results[0].plot()

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ================== FOOTER ==================
st.markdown("---")
st.caption(
    "YOLOv8 Custom Model & Dataset Piring dan Mangkok Object Detection - "
    "Muhammad Luthfan Lazuardi (24060122120010)"
)
