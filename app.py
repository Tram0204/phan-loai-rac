import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import hashlib
from datetime import datetime
import io
import base64
import cv2

# Tắt các warning của TensorFlow
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

# Đảm bảo set_page_config là dòng đầu tiên
st.set_page_config(
    page_title="Phân loại rác thông minh AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #00C9FF;
    }
    
    .drag-drop-area {
        border: 2px dashed #00C9FF;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    
    .dark-mode {
        background-color: #1e1e1e;
        color: white;
    }
    
    .recycling-tip {
        background: #e8f5e8;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# === Thông tin lớp rác với tips tái chế ===
LABELS_INFO = {
    "cardboard": {
        "vi": "Bìa cứng",
        "info": "Bìa cứng như hộp giấy, hộp đựng pizza...",
        "image": "resources/cardboard.jpg",
        "recycling_tips": [
            "Tháo bỏ băng dính và kim ghim trước khi tái chế",
            "Làm phẳng hộp để tiết kiệm không gian",
            "Không tái chế hộp pizza có dính dầu mỡ",
            "Có thể tái chế thành giấy mới hoặc bao bì khác"
        ],
        "collection_points": "Thùng rác tái chế màu xanh lá",
        "color": "#8B4513"
    },
    "glass": {
        "vi": "Thủy tinh",
        "info": "Chai lọ thủy tinh, mảnh vỡ thủy tinh...",
        "image": "resources/glass.jpg",
        "recycling_tips": [
            "Rửa sạch trước khi bỏ vào thùng tái chế",
            "Tháo nắp kim loại hoặc nhựa",
            "Không trộn với gốm sứ",
            "Thủy tinh có thể tái chế vô hạn lần"
        ],
        "collection_points": "Thùng rác tái chế màu xanh dương",
        "color": "#00CED1"
    },
    "metal": {
        "vi": "Kim loại",
        "info": "Lon nhôm, hộp thiếc...",
        "image": "resources/metal.jpg",
        "recycling_tips": [
            "Rửa sạch thức ăn còn sót lại",
            "Không cần tháo nhãn dán",
            "Lon nhôm có thể tái chế thành lon mới trong 60 ngày",
            "Kim loại tiết kiệm 95% năng lượng khi tái chế"
        ],
        "collection_points": "Thùng rác tái chế màu xám",
        "color": "#C0C0C0"
    },
    "paper": {
        "vi": "Giấy",
        "info": "Giấy báo, giấy in, sách cũ...",
        "image": "resources/paper.jpg",
        "recycling_tips": [
            "Tháo bỏ kẹp giấy và ghim bấm",
            "Không tái chế giấy có dính băng dính",
            "Giấy ướt không thể tái chế",
            "1 tấn giấy tái chế tiết kiệm 17 cây"
        ],
        "collection_points": "Thùng rác tái chế màu xanh lá",
        "color": "#228B22"
    },
    "plastic": {
        "vi": "Nhựa",
        "info": "Chai nhựa, bao bì nhựa...",
        "image": "resources/plastic.jpg",
        "recycling_tips": [
            "Kiểm tra mã số tái chế trên đáy chai",
            "Rửa sạch và tháo nắp",
            "Nhựa số 1,2,5 dễ tái chế nhất",
            "Tránh nhựa đen vì khó phân loại"
        ],
        "collection_points": "Thùng rác tái chế màu vàng",
        "color": "#FFD700"
    },
    "trash": {
        "vi": "Rác thải khác",
        "info": "Rác hữu cơ, tã giấy, gói bim bim...",
        "image": "resources/trash.jpg",
        "recycling_tips": [
            "Phân loại rác hữu cơ để làm phân compost",
            "Tã giấy cần bỏ vào túi riêng",
            "Rác thực phẩm có thể ủ phân",
            "Giảm thiểu rác thải bằng cách tái sử dụng"
        ],
        "collection_points": "Thùng rác thông thường màu đen",
        "color": "#800080"
    }
}

# === Initialize session state ===
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.7
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("trashnet_model.keras")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

model = load_model()
class_names = list(LABELS_INFO.keys())

# === Utility Functions ===
def get_image_hash(image):
    """Tạo hash cho ảnh để phát hiện trùng lặp"""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return hashlib.md5(image_bytes.getvalue()).hexdigest()

def auto_rotate_image(image):
    """Tự động xoay ảnh dựa trên dữ liệu EXIF"""
    try:
        return ImageOps.exif_transpose(image)
    except:
        return image

def enhance_image(image, brightness=1.0, contrast=1.0):
    """Tăng cường độ sáng và độ tương phản của ảnh"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image

def process_image(image: Image.Image, enhance_params=None):
    """Xử lý ảnh với các tùy chỉnh tăng cường"""
    # Tự động xoay
    image = auto_rotate_image(image)
    
    # Tăng cường nếu có tham số
    if enhance_params:
        image = enhance_image(image, 
                            brightness=enhance_params['brightness'],
                            contrast=enhance_params['contrast'])
    
    img = image.convert("RGB").resize((224, 224))
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array / 255.0
    return np.expand_dims(array, axis=0)

def create_confidence_chart(predictions, labels):
    """Tạo biểu đồ tương tác độ tin cậy"""
    fig = go.Figure(data=[
        go.Bar(
            x=predictions,
            y=labels,
            orientation='h',
            marker_color=['#ff6b6b' if p < 0.7 else '#4ecdc4' for p in predictions],
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Độ tin cậy dự đoán",
        xaxis_title="Xác suất",
        yaxis_title="Loại rác",
        height=400,
        showlegend=False
    )
    
    return fig

# === Main Interface ===
def main():
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Phân loại rác thông minh</h1>
        <p>Ứng dụng AI tiên tiến giúp phân loại rác và bảo vệ môi trường</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Dark mode toggle
        if st.checkbox("🌙 Chế độ tối", value=st.session_state.dark_mode):
            st.session_state.dark_mode = True
            st.markdown('<style>body {background-color: #1e1e1e; color: white;}</style>', 
                       unsafe_allow_html=True)
    
        # Image enhancement
        st.subheader("📸 Tăng cường ảnh")
        brightness = st.slider("Độ sáng", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Độ tương phản", 0.5, 2.0, 1.0, 0.1)
        
        # Camera capture
        st.subheader("📷 Chụp ảnh trực tiếp")
        camera_image = st.camera_input("Chụp ảnh rác")
        
        # Clear history
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.history_data = []
            st.session_state.processed_images = {}
            st.success("Đã xóa lịch sử!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Tải lên hình ảnh")
        
        # Drag and drop area
        st.markdown("""
        <div class="drag-drop-area">
            <h3>🎯 Kéo thả hoặc click để tải ảnh</h3>
            <p>Hỗ trợ JPG, PNG, JPEG • Tối đa 200MB mỗi ảnh</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Chọn nhiều ảnh", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        # Process camera image
        if camera_image:
            uploaded_files = [camera_image] + (uploaded_files or [])
    
    with col2:
        st.subheader("📊 Thống kê nhanh")
        
        # Quick stats
        # Quick stats
        total_images = len(st.session_state.history_data)
        if total_images > 0:
            df = pd.DataFrame(st.session_state.history_data)
            most_common = df['Loại rác'].mode().iloc[0] if not df.empty else "Chưa có"
    
            try:
                df['Độ chính xác (%)'] = df['Độ chính xác (%)'].str.rstrip('%').astype(float)
                avg_confidence = df['Độ chính xác (%)'].dropna().mean()
                avg_display = f"{avg_confidence:.1f}%"
            except Exception as e:
                avg_display = "Không xác định"
                st.warning(f"Lỗi xử lý độ chính xác trung bình: {e}")

            st.metric("Tổng số ảnh đã xử lý", total_images)
            st.metric("Loại rác phổ biến", most_common)
            st.metric("Độ chính xác trung bình", avg_display)
        else:
            st.info("Chưa có dữ liệu thống kê")
    # Process uploaded images
    if uploaded_files:
        st.subheader("🔍 Kết quả phân loại")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        duplicate_count = 0
        low_confidence_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Đang xử lý ảnh {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Load and process image
            img = Image.open(uploaded_file)
            img_hash = get_image_hash(img)
            
            # Check for duplicates
            if img_hash in st.session_state.processed_images:
                duplicate_count += 1
                st.warning(f"⚠️ Ảnh {uploaded_file.name} đã được xử lý trước đó!")
                continue
            
            # Process image
            enhance_params = {'brightness': brightness, 'contrast': contrast}
            input_tensor = process_image(img, enhance_params)
            
            # Predict
            with st.spinner("Đang phân tích..."):
                prediction = model.predict(input_tensor, verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                predicted_label = class_names[predicted_idx]
                confidence = prediction[predicted_idx]
            
            # Store result
            vi_label = LABELS_INFO[predicted_label]["vi"]
            description = LABELS_INFO[predicted_label]["info"]
            
            result = {
                'file': uploaded_file,
                'image': img,
                'label': predicted_label,
                'vi_label': vi_label,
                'confidence': confidence,
                'description': description,
                'prediction': prediction,
                'hash': img_hash
            }
            results.append(result)
            
            # Check confidence
            if confidence < st.session_state.confidence_threshold:
                low_confidence_count += 1
            
            # Store in session
            st.session_state.processed_images[img_hash] = result
            st.session_state.history_data.append({
                "Tên ảnh": uploaded_file.name,
                "Loại rác": vi_label,
                "Độ chính xác (%)": f"{confidence*100:.2f}",
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Show warnings
        if duplicate_count > 0:
            st.warning(f"⚠️ Phát hiện {duplicate_count} ảnh trùng lặp!")
        
        if low_confidence_count > 0:
            st.warning(f"⚠️ {low_confidence_count} ảnh có độ tin cậy thấp (<{st.session_state.confidence_threshold*100:.0f}%)")
        
        # Display results
        for result in results:
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h4>📋 Kết quả: {result['file'].name}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.image(result['image'], caption="Ảnh gốc", use_container_width=True)
                
                with col2:
                    st.markdown(f"**🏷️ Loại rác:** `{result['vi_label']}`")
                    st.markdown(f"**📊 Độ chính xác:** `{result['confidence']*100:.2f}%`")
                    st.markdown(f"**📝 Mô tả:** {result['description']}")
                    
                    # Confidence warning
                    if result['confidence'] < st.session_state.confidence_threshold:
                        st.markdown("""
                        <div class="warning-card">
                            ⚠️ Độ tin cậy thấp! Kiểm tra lại kết quả.
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    # Interactive confidence chart
                    labels = [LABELS_INFO[c]["vi"] for c in class_names]
                    fig = create_confidence_chart(result['prediction'], labels)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recycling tips
                st.markdown(f"""
                <div class="recycling-tip">
                    <h5>♻️ Hướng dẫn tái chế {result['vi_label']}:</h5>
                </div>
                """, unsafe_allow_html=True)
                
                tips = LABELS_INFO[result['label']]['recycling_tips']
                for tip in tips:
                    st.markdown(f"• {tip}")
                
                collection_point = LABELS_INFO[result['label']]['collection_points']
                st.info(f"📍 **Điểm thu gom:** {collection_point}")
                
                st.markdown("---")
    
    # Analytics Dashboard
    if st.session_state.history_data:
        st.subheader("📈 Phân tích thống kê")
        
        df = pd.DataFrame(st.session_state.history_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution chart
            trash_counts = df['Loại rác'].value_counts()
            fig = px.pie(
                values=trash_counts.values,
                names=trash_counts.index,
                title="Phân bố loại rác"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of trash types by confidence
        if len(df) > 5:
            heatmap_data = df.pivot_table(
                values='Confidence',
                index='Loại rác',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(
                heatmap_data.values.reshape(1, -1),
                x=heatmap_data.index,
                aspect='auto',
                title="Độ tin cậy trung bình theo loại rác"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("📤 Xuất dữ liệu")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📊 Tải tệp CSV",
                data=csv,
                file_name=f"phan_loai_rac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Phân loại rác')
            
            st.download_button(
                "📈 Tải tệp Excel",
                data=excel_buffer.getvalue(),
                file_name=f"phan_loai_rac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        
        # History table
        st.subheader("📝 Lịch sử chi tiết")
        
        # Search and filter
        search_term = st.text_input("🔍 Tìm kiếm trong lịch sử:")
        if search_term:
            filtered_df = df[df['Tên ảnh'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df
        
        # Display with formatting
        filtered_df['Độ chính xác (%)'] = pd.to_numeric(filtered_df['Độ chính xác (%)'], errors='coerce')

        st.dataframe(
            filtered_df.style.format({'Độ chính xác (%)': '{:.2f}%'}),
            use_container_width=True
)

    
    # Educational section
    with st.expander("📚 Tìm hiểu về phân loại rác"):
        st.markdown("### 🌍 Tại sao phân loại rác quan trọng?")
        st.markdown("""
        - **Bảo vệ môi trường**: Giảm ô nhiễm đất, nước và không khí
        - **Tiết kiệm tài nguyên**: Tái chế giúp tiết kiệm nguyên liệu thô
        - **Giảm khí thải**: Giảm phát thải CO2 và khí nhà kính
        - **Kinh tế tuần hoàn**: Tạo ra giá trị từ chất thải
        """)
        
        st.markdown("### ♻️ Hướng dẫn chi tiết từng loại rác:")
        
        for label, info in LABELS_INFO.items():
            with st.container():
                st.markdown(f"#### {info['vi']}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(info['image']):
                        st.image(info['image'], caption=f"Hình minh họa: {info['vi']}", width=150)
                
                with col2:
                    st.markdown(f"**Mô tả:** {info['info']}")
                    st.markdown(f"**Điểm thu gom:** {info['collection_points']}")
                    
                    st.markdown("**Cách tái chế:**")
                    for tip in info['recycling_tips']:
                        st.markdown(f"• {tip}")
                
                st.markdown("---")
    
    # Footer
    st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>📬 Liên hệ</h4>
    <p>Nếu bạn có thắc mắc, góp ý hoặc muốn hợp tác, vui lòng liên hệ:</p>
    <p>📧 Email: <a href="mailto:fftt0519@gmail.com">fftt0519@gmail.com</a></p>
    <p>📞 Điện thoại: 0339336571</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
