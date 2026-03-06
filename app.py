import streamlit as st
import cv2
import numpy as np
from tree_module.analyze import analyze_forest

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

h1 {
    color: #22c55e;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: #111827;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 EcoVision AI")
st.caption("AI Powered Forest Monitoring & Analytics")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Forest Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    cv2.imwrite("temp.jpg", image)

    with st.spinner("Running AI Forest Analysis..."):
        result = analyze_forest("temp.jpg")

    st.divider()

    st.subheader("AI Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("outputs/tree_detections.png", caption="Tree Detection")

    with col2:
        st.image("outputs/tree_density_heatmap.png", caption="Tree Density Heatmap")

    with col3:
        st.image("outputs/vegetation_health_map.png", caption="Vegetation Health (NDVI)")

    st.divider()

    st.subheader("Forest Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌳 Tree Count",
        result["tree_count"]
    )

    c2.metric(
        "📊 Tree Density",
        f'{result["tree_density"]:.2f} trees/km²'
    )

    c3.metric(
        "🌱 Forest Health Score",
        result["forest_health_score"]
    )

    c4.metric(
        "📏 Avg Tree Spacing",
        f'{result["avg_tree_spacing"]:.2f} m'
    )

    st.divider()

    st.subheader("Species Classification")

    st.info("Species segmentation module coming soon (integration in progress)")