import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go


IMG_SIZE = 124
NUM_CLASSES = 8
CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']


st.set_page_config(
    page_title="Retinal Disease Classification",
    page_icon="👁",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .model-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .predicted-class {
        font-size: 1.8rem;
        font-weight: bold;
        color: #27ae60;
        margin: 0.5rem 0;
    }
    .confidence {
        font-size: 1.1rem;
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">👁 Retinal Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-class Classification Using OCT Images</div>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        base_model = tf.keras.models.load_model('base_cnn.keras')
        mobilenet_model = tf.keras.models.load_model('custom_mobilenetv2_tuned_classifier.keras')
        return base_model, mobilenet_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure both model files are in the same directory as this script.")
        return None, None

def preprocess_image(image):
    """Preprocess image: resize and normalize"""
    # Resize image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array
    img_array = np.array(img)
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, img_array):
    """Make prediction using the model"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return predicted_class, confidence, predictions[0]

def create_probability_chart(probabilities, model_name):
    """Create an interactive bar chart for class probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=probabilities * 100,
            marker_color=['#27ae60' if p == max(probabilities) else '#3498db' for p in probabilities],
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'{model_name} - Class Probabilities',
        xaxis_title='Disease Class',
        yaxis_title='Probability (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

# Load models
base_model, mobilenet_model = load_models()
if base_model is not None and mobilenet_model is not None:
    st.success("✅ Models loaded successfully!")
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a retinal OCT image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal OCT scan image for disease classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded OCT Image', use_container_width=True)
        
        st.markdown("---")
        
        # Preprocess button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner('Processing image and making predictions...'):
                # Preprocess image
                preprocessed_img = preprocess_image(image)
                
                # Make predictions with both models
                base_class, base_conf, base_probs = predict_disease(base_model, preprocessed_img)
                mobile_class, mobile_conf, mobile_probs = predict_disease(mobilenet_model, preprocessed_img)
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Display results side by side
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown('<div class="model-name">🔹 Base CNN Model</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="predicted-class">Prediction: {base_class}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence">Confidence: {base_conf:.2f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability chart
                    fig_base = create_probability_chart(base_probs, "Base CNN Model")
                    st.plotly_chart(fig_base, use_container_width=True)
                
                with col_right:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown('<div class="model-name">🔹 MobileNetV2 Model</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="predicted-class">Prediction: {mobile_class}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence">Confidence: {mobile_conf:.2f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability chart
                    fig_mobile = create_probability_chart(mobile_probs, "MobileNetV2 Model")
                    st.plotly_chart(fig_mobile, use_container_width=True)
                
                # Class descriptions
                st.markdown("---")
                st.markdown("### 📋 Disease Class Information")
                
                class_info = {
                    'AMD': 'Age-related Macular Degeneration',
                    'CNV': 'Choroidal Neovascularization',
                    'CSR': 'Central Serous Retinopathy',
                    'DME': 'Diabetic Macular Edema',
                    'DR': 'Diabetic Retinopathy',
                    'DRUSEN': 'Drusen deposits',
                    'MH': 'Macular Hole',
                    'NORMAL': 'Normal Retina'
                }
                
                cols = st.columns(4)
                for idx, (abbr, full_name) in enumerate(class_info.items()):
                    with cols[idx % 4]:
                        st.info(f"{abbr}: {full_name}")
                
    else:
        st.info("👆 Please upload a retinal OCT image to begin analysis")
        
        # Example information
        with st.expander("ℹ️ About this application"):
            st.write("""
This application uses two deep learning models to classify retinal diseases from OCT images:
            
            1. Base CNN Model: A custom convolutional neural network
            2. MobileNetV2 Model: A fine-tuned MobileNetV2 architecture
            
            Supported Disease Classes:
            - AMD (Age-related Macular Degeneration)
            - CNV (Choroidal Neovascularization)
            - CSR (Central Serous Retinopathy)
            - DME (Diabetic Macular Edema)
            - DR (Diabetic Retinopathy)
            - DRUSEN (Drusen deposits)
            - MH (Macular Hole)
            - NORMAL (Normal Retina)
            
            Image Requirements:
            - Format: PNG, JPG, or JPEG
            - The image will be automatically resized to 124x124 pixels
            - Images are normalized during preprocessing
            """)

else:
    st.error("Failed to load models. Please check if the model files exist in the correct location.")
    st.markdown("""
    Required files:
    - base_cnn_model.keras
    - custom_mobilenetv2_tuned_classifer.keras
    
    Place these files in the same directory as this script.
    """)
