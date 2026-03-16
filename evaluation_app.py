# evaluation_app.py

# SCROLL FIX AT VERY TOP
import streamlit as st
import streamlit.components.v1 as components
# Replace the scroll section with this:
st.components.v1.html("""
    <script>
        // Get the Streamlit app container and scroll it
        var app = window.parent.document.querySelector('.stApp');
        if (app) {
            app.scrollTop = 0;
        }
        // Also try the main scroll container
        var main = window.parent.document.querySelector('main');
        if (main) {
            main.scrollTop = 0;
        }
        // And the body
        window.parent.document.body.scrollTop = 0;
        window.parent.document.documentElement.scrollTop = 0;
    </script>
""", height=0)

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import sys
import os

# Try to import cv2, fallback to None if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Leaf Disease Evaluation",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .shap-card {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #f57c00;
        margin: 0.5rem 0;
    }
    .sample-info {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .heatmap-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #555;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Leaf Disease Detection Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid CNN-ViT Ensemble Model Assessment</div>', unsafe_allow_html=True)

# Debug info in sidebar
with st.sidebar.expander("Debug Info"):
    st.write(f"Python: {sys.executable}")
    st.write(f"CV2 Available: {CV2_AVAILABLE}")

def load_samples():
    """Load evaluation samples with ensemble-only SHAP support"""
    try:
        possible_paths = [
            'evaluation_samples_with_shap.pkl',
            'evaluation_samples.pkl',
            './evaluation_samples_with_shap.pkl',
            './evaluation_samples.pkl', 
            '../evaluation_samples_with_shap.pkl',
            '../evaluation_samples.pkl',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    samples = pickle.load(f)
                
                has_shap_count = sum(1 for s in samples 
                                   if 'shap' in s.get('explanations', {}) 
                                   and s['explanations']['shap'])
                
                st.sidebar.success(f"Loaded {len(samples)} samples")
                if has_shap_count > 0:
                    st.sidebar.success(f"{has_shap_count} with SHAP")
                
                return samples
        
        st.error("""
        **Evaluation samples not found!** 
        
        Please generate samples first by running the sample generation code.
        """)
        return []
    except Exception as e:
        st.error(f"Error loading samples: {str(e)}")
        return []

# Initialize session state
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = 0
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Sample Evaluation"

# Sidebar Navigation
st.sidebar.markdown("## Navigation")

nav_options = {
    "Sample Evaluation": "Samples",
    "Upload New Image": "Upload",
    "View Results": "Results"
}

for mode, label in nav_options.items():
    if st.sidebar.button(
        label, 
        key=f"nav_{mode}",
        use_container_width=True,
        type="primary" if st.session_state.app_mode == mode else "secondary"
    ):
        st.session_state.app_mode = mode
        st.rerun()

st.sidebar.markdown("---")
samples = load_samples()

def get_shap_image(sample):
    """Extract SHAP visualization image from sample data."""
    try:
        explanations = sample.get('explanations', {})
        shap_container = explanations.get('shap', {})
        
        if not shap_container:
            return None
        
        if 'ensemble' in shap_container:
            ensemble_data = shap_container['ensemble']
            if isinstance(ensemble_data, np.ndarray):
                return ensemble_data
            if isinstance(ensemble_data, dict):
                if 'visualization' in ensemble_data:
                    return ensemble_data['visualization']
                if 'image' in ensemble_data:
                    return ensemble_data['image']
                if 'overlay' in ensemble_data:
                    return ensemble_data['overlay']
        
        if isinstance(shap_container, np.ndarray):
            return shap_container
            
        return None
        
    except Exception as e:
        st.error(f"Error extracting SHAP: {str(e)}")
        return None

def display_explanations_inline(sample):
    """Display all visualizations in aligned rows"""
    explanations = sample.get('explanations', {})
    
    # ROW 1: Original Image + Grad-CAM Heatmaps (all same height)
    st.markdown('<div class="section-header">Model Attention Maps</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    # Column 1: Original Image
    with cols[0]:
        st.markdown('<div class="heatmap-label">Original Image</div>', unsafe_allow_html=True)
        if sample.get('image_array') is not None:
            st.image(sample['image_array'], use_container_width=True)
    
    # Column 2: ResNet50
    with cols[1]:
        st.markdown('<div class="heatmap-label">ResNet50</div>', unsafe_allow_html=True)
        if explanations.get('resnet50_gradcam') and explanations['resnet50_gradcam'].get('overlay') is not None:
            st.image(explanations['resnet50_gradcam']['overlay'], use_container_width=True)
        else:
            st.info("No data")
    
    # Column 3: EfficientNet
    with cols[2]:
        st.markdown('<div class="heatmap-label">EfficientNet</div>', unsafe_allow_html=True)
        if explanations.get('efficientnet_gradcam') and explanations['efficientnet_gradcam'].get('overlay') is not None:
            st.image(explanations['efficientnet_gradcam']['overlay'], use_container_width=True)
        else:
            st.info("No data")
    
    # Column 4: ViT
    with cols[3]:
        st.markdown('<div class="heatmap-label">ViT</div>', unsafe_allow_html=True)
        if explanations.get('vit_attention') and explanations['vit_attention'].get('overlay') is not None:
            st.image(explanations['vit_attention']['overlay'], use_container_width=True)
        else:
            st.info("No data")
    
    # ROW 2: SHAP Explanation (full width)
    st.markdown('<div class="section-header">Ensemble SHAP Explanation</div>', unsafe_allow_html=True)
    
    shap_img = get_shap_image(sample)
    
    if shap_img is not None:
        st.image(shap_img, use_container_width=True, caption="Original | Importance | Positive | Negative | Overlay")
        
        st.markdown("""
        <div class="shap-card">
        <strong>SHAP Guide:</strong> 
        <strong>Importance</strong> (hot = critical pixels) | 
        <strong>Positive</strong> (supports prediction) | 
        <strong>Negative</strong> (opposes prediction) | 
        <strong>Overlay</strong> (blended on image)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No SHAP data available for this sample.")
    
    # Quality Metrics (compact row)
    st.markdown('<div class="section-header">Attention Quality Metrics</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        if explanations.get('resnet50_gradcam') and explanations['resnet50_gradcam'].get('heatmap') is not None:
            heatmap = explanations['resnet50_gradcam']['heatmap']
            focus = np.mean(heatmap > 0.5)
            st.metric("ResNet Focus", f"{focus:.1%}", help="% of image with high attention")
    
    with metric_cols[1]:
        if explanations.get('efficientnet_gradcam') and explanations['efficientnet_gradcam'].get('heatmap') is not None:
            heatmap = explanations['efficientnet_gradcam']['heatmap']
            focus = np.mean(heatmap > 0.5)
            st.metric("EffNet Focus", f"{focus:.1%}", help="% of image with high attention")
    
    with metric_cols[2]:
        if explanations.get('vit_attention') and explanations['vit_attention'].get('heatmap') is not None:
            heatmap = explanations['vit_attention']['heatmap']
            focus = np.mean(heatmap > 0.5)
            st.metric("ViT Focus", f"{focus:.1%}", help="% of image with high attention")
    
    with metric_cols[3]:
        st.metric("SHAP", "Yes" if shap_img is not None else "No")

# ==================== SAMPLE EVALUATION ====================
if st.session_state.app_mode == "Sample Evaluation":
    st.header("Sample Evaluation")
    
    if samples:
        # Scroll to top using hash navigation
        st.markdown("""
            <script>
                window.location.hash = 'top';
            </script>
        """, unsafe_allow_html=True)
        
        # Sample selector
        sample_options = []
        for i, sample in enumerate(samples):
            has_shap = get_shap_image(sample) is not None
            shap_indicator = " *" if has_shap else ""
            true_label = sample.get('true_label', 'Unknown')
            pred_label = sample.get('ensemble_prediction', {}).get('class', 'Unknown')
            sample_options.append(f"Sample {i+1}: {true_label} -> {pred_label}{shap_indicator}")
        
        # Controls
        col_select, col_random = st.columns([4, 1])
        with col_select:
            selected_option = st.selectbox(
                "Select sample:",
                range(len(samples)),
                index=st.session_state.current_sample,
                format_func=lambda x: sample_options[x]
            )
        
        with col_random:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Random", use_container_width=True):
                import random
                st.session_state.current_sample = random.randint(0, len(samples) - 1)
                st.rerun()
        
        st.session_state.current_sample = selected_option
        current_sample = samples[selected_option]
        
        # Info banner
        has_shap = get_shap_image(current_sample) is not None
        true_label = current_sample.get('true_label', 'Unknown')
        pred_class = current_sample.get('ensemble_prediction', {}).get('class', 'Unknown')
        confidence = current_sample.get('ensemble_prediction', {}).get('confidence', 0)
        
        st.markdown(f"""
        <div class="sample-info">
            <strong>Sample {selected_option + 1} of {len(samples)}</strong> | 
            True: <code>{true_label}</code> -> 
            Pred: <code>{pred_class}</code> |
            Conf: <strong>{confidence:.3f}</strong>
            {' | SHAP' if has_shap else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # === ALIGNED VISUALIZATIONS ===
        display_explanations_inline(current_sample)
        
        # === PREDICTION INFO (compact sidebar style) ===
        st.markdown("---")
        
        info_cols = st.columns([1, 2])
        
        with info_cols[0]:
            st.markdown('<div class="section-header">Prediction Details</div>', unsafe_allow_html=True)
            conf_color = "#4caf50" if confidence > 0.8 else "#ff9800" if confidence > 0.5 else "#f44336"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {conf_color};">
                <strong>True:</strong> {true_label}<br>
                <strong>Pred:</strong> {pred_class}<br>
                <strong>Conf:</strong> <span style="color: {conf_color};">{confidence:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with info_cols[1]:
            st.markdown('<div class="section-header">Individual Model Predictions</div>', unsafe_allow_html=True)
            ind_preds = current_sample.get('individual_predictions', {})
            pred_cols = st.columns(len(ind_preds))
            for i, (model_name, pred) in enumerate(ind_preds.items()):
                with pred_cols[i]:
                    display_name = model_name.replace('_', ' ').title()
                    conf = pred.get('confidence', 0)
                    st.metric(display_name, f"{pred.get('class', 'N/A')}", f"{conf:.3f}")
        
        # === EVALUATION FORM ===
        st.markdown("---")
        st.subheader("Expert Evaluation")
        
        with st.form(f"eval_form_{selected_option}"):
            col_f1, col_f2 = st.columns([1, 1])
            
            with col_f1:
                is_correct = "Correct" if true_label == pred_class else "Incorrect"
                correct = st.radio(
                    "Classification",
                    ["Correct", "Incorrect", "Partially Correct"],
                    index=["Correct", "Incorrect", "Partially Correct"].index(is_correct) if is_correct in ["Correct", "Incorrect"] else 2,
                    horizontal=True
                )
            
            with col_f2:
                if has_shap:
                    st.success("SHAP available")
                else:
                    st.info("No SHAP")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                heatmap_quality = st.slider("Grad-CAM Quality", 1, 5, 3)
            with col_s2:
                shap_quality = st.slider("SHAP Quality", 1, 5, 3) if has_shap else st.empty()
            with col_s3:
                usefulness = st.slider("Usefulness", 1, 5, 3)
            
            comments = st.text_area(
                "Comments",
                placeholder="Expert observations...",
                height=80
            )
            
            submitted = st.form_submit_button("Submit Evaluation", type="primary", use_container_width=True)
            
            if submitted:
                evaluation = {
                    'sample_id': selected_option,
                    'sample_info': {
                        'true_label': true_label,
                        'predicted_label': pred_class,
                        'confidence': confidence,
                        'has_shap': has_shap
                    },
                    'correct': correct,
                    'heatmap_quality': heatmap_quality,
                    'shap_quality': shap_quality if has_shap else None,
                    'usefulness': usefulness,
                    'comments': comments,
                    'timestamp': str(np.datetime64('now'))
                }
                st.session_state.evaluations.append(evaluation)
                st.success("Evaluation submitted!")
                st.balloons()
        
        # Navigation
        st.markdown("---")
        nav_cols = st.columns([1, 1, 1, 1, 2])
        
        with nav_cols[0]:
            if st.button("Prev", disabled=(selected_option == 0), use_container_width=True):
                st.session_state.current_sample = selected_option - 1
                st.rerun()
        
        with nav_cols[1]:
            if st.button("Next", disabled=(selected_option >= len(samples) - 1), use_container_width=True):
                st.session_state.current_sample = selected_option + 1
                st.rerun()
        
        with nav_cols[2]:
            jump_num = st.number_input("Go to", min_value=1, max_value=len(samples), value=selected_option + 1, label_visibility="collapsed")
        
        with nav_cols[3]:
            if st.button("Jump", use_container_width=True):
                st.session_state.current_sample = jump_num - 1
                st.rerun()
        
        with nav_cols[4]:
            st.progress((selected_option + 1) / len(samples), text=f"Progress: {selected_option + 1}/{len(samples)}")
    
    else:
        st.warning("No samples loaded. Generate evaluation samples first.")

# ==================== UPLOAD MODE ====================
elif st.session_state.app_mode == "Upload New Image":
    st.header("Upload Image")
    st.info("Live inference mode - coming soon")
    
    uploaded = st.file_uploader("Select leaf image:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded", width=400)
        st.warning("Live model inference not yet implemented")

# ==================== RESULTS MODE ====================
elif st.session_state.app_mode == "View Results":
    st.header("Evaluation Results")
    
    if st.session_state.evaluations:
        total = len(st.session_state.evaluations)
        
        correct = sum(1 for e in st.session_state.evaluations if e['correct'] == 'Correct')
        partial = sum(1 for e in st.session_state.evaluations if e['correct'] == 'Partially Correct')
        incorrect = sum(1 for e in st.session_state.evaluations if e['correct'] == 'Incorrect')
        
        avg_heatmap = np.mean([e['heatmap_quality'] for e in st.session_state.evaluations])
        avg_useful = np.mean([e['usefulness'] for e in st.session_state.evaluations])
        shap_evals = sum(1 for e in st.session_state.evaluations if e['sample_info'].get('has_shap'))
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Total", total)
        mcol2.metric("Accuracy", f"{correct/total*100:.1f}%")
        mcol3.metric("Avg Heatmap", f"{avg_heatmap:.1f}/5")
        mcol4.metric("Avg Usefulness", f"{avg_useful:.1f}/5")
        
        if shap_evals > 0:
            avg_shap = np.mean([e['shap_quality'] for e in st.session_state.evaluations if e['shap_quality'] is not None])
            st.metric("Avg SHAP Quality", f"{avg_shap:.1f}/5")
        
        st.subheader("Evaluation History")
        for i, ev in enumerate(reversed(st.session_state.evaluations)):
            idx = len(st.session_state.evaluations) - i
            with st.expander(f"#{idx}: {ev['sample_info']['true_label']} -> {ev['sample_info']['predicted_label']} ({ev['correct']})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Confidence:** {ev['sample_info']['confidence']:.3f}")
                    st.write(f"**Heatmap:** {ev['heatmap_quality']}/5")
                with c2:
                    st.write(f"**Usefulness:** {ev['usefulness']}/5")
                    if ev['shap_quality']:
                        st.write(f"**SHAP:** {ev['shap_quality']}/5")
                if ev['comments']:
                    st.write(f"**Comments:** {ev['comments']}")
        
        st.markdown("---")
        export_data = {
            'summary': {
                'total': total,
                'correct': correct,
                'partial': partial,
                'incorrect': incorrect,
                'accuracy': correct/total if total > 0 else 0,
                'avg_heatmap': avg_heatmap,
                'avg_usefulness': avg_useful,
                'shap_evaluations': shap_evals
            },
            'evaluations': st.session_state.evaluations
        }
        
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            f"eval_results_{np.datetime64('now')}.json",
            use_container_width=True
        )
    else:
        st.info("No evaluations yet. Start evaluating samples!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Guide:**
- * = SHAP available
- Navigate with Prev/Next or Jump
""")