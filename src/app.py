import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import json

# ==========================================
# PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="FSLSM Predictor (Panel Dashboard)", layout="wide", initial_sidebar_state="expanded")

if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = None

# ==========================================
# DATA LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_models_and_data():
    """Load SOTA models, dataset, and metadata with caching"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'sota_semi_supervised_models.joblib')
    data_path = os.path.join(base_dir, 'data', 'data_fs1.csv')
    arch_img_path = os.path.join(base_dir, 'docs', 'cpl_process_diagram.png')
    
    try:
        # Load custom module so unpickling works
        import sys
        if os.path.join(base_dir, 'src') not in sys.path:
            sys.path.insert(0, os.path.join(base_dir, 'src'))
        import model_definitions
        import curriculum_self_training
        import torch # Needed for PyTorch tabnet/KAN
        
        models = joblib.load(model_path)
        df = pd.read_csv(data_path)
        
        return models, df, arch_img_path
    except Exception as e:
        st.error(f"Error loading models or data: {str(e)}")
        return None, None, None

models_dict, df_raw, arch_img_path = load_models_and_data()

# ==========================================
# PREDICTION ENGINE
# ==========================================
def predict_learning_style(input_data, models):
    if not models:
        return None
        
    feature_order = [
        'T_image', 'T_video', 'T_read', 'T_audio', 'T_hierarchies', 'T_powerpoint', 
        'T_concrete', 'T_result', 'N_standard_questions_correct', 'N_msgs_posted', 
        'T_solve_excercise', 'N_group_discussions', 'Skipped_los', 'N_next_button_used', 
        'T_spent_in_session', 'N_questions_on_details', 'N_questions_on_outlines'
    ]
    
    # Map sliders to raw features
    mapped_data = {
        'T_image': input_data['diagram_view_time'],
        'T_video': input_data['video_watch_time'],
        'T_read': input_data['reading_time'],
        'T_audio': input_data['theoretical_discussions'], 
        'T_hierarchies': input_data['big_picture_focus'] / 4, 
        'T_powerpoint': input_data['visual_content_engagement'],
        'T_concrete': input_data['practical_exercises'],
        'T_result': input_data['step_by_step_completion'] / 4,
        'N_standard_questions_correct': input_data['pattern_recognition'],
        'N_msgs_posted': input_data['messages_posted'],
        'T_solve_excercise': input_data['hands_on_activities'],
        'N_group_discussions': input_data['group_discussions'],
        'Skipped_los': 5,
        'N_next_button_used': input_data['linear_progression'] * 5,
        'T_spent_in_session': 30,
        'N_questions_on_details': input_data['detail_orientation'] * 2,
        'N_questions_on_outlines': input_data['holistic_understanding'] * 2
    }
    
    X_raw = pd.DataFrame([mapped_data])[feature_order]
    
    # Apply FCM logic used in training
    import skfuzzy as fuzz
    try:
        data_for_fcm = X_raw.values.astype(np.float64).T
        cntr, u, _, _, _, _, _ = fuzz.cmeans(data_for_fcm, c=8, m=2.0, error=0.005, maxiter=100)
        fcm_features = pd.DataFrame(u.T, columns=[f'FCM_M_{i+1}' for i in range(8)], index=X_raw.index)
        X_input = pd.concat([X_raw, fcm_features], axis=1)
    except Exception:
        # Fallback if FCM fails for single row
        X_input = pd.concat([X_raw, pd.DataFrame(np.zeros((1, 8)), columns=[f'FCM_M_{i+1}' for i in range(8)])], axis=1)
    
    results = {}
    
    for dim in ['visual_verbal', 'active_reflective', 'sensing_intuitive', 'sequential_global']:
        if dim not in models:
            continue
            
        model_info = models[dim]
        model = model_info['model']
        
        # Format input (PyTorch models need float32)
        X_val = X_input.values.astype(np.float32)
        
        # Predict Proba
        try:
            probabilities = model.predict_proba(X_val)[0]
            prob_class_1 = probabilities[1]
            prob_class_0 = probabilities[0]
            
            # Label mapping based on training:
            # visual_verbal: 1=Visual, 0=Verbal
            # active_reflective: 1=Active, 0=Reflective
            # sensing_intuitive: 1=Sensing, 0=Intuitive
            # sequential_global: 1=Sequential, 0=Global
            
            label_1 = dim.split('_')[0].capitalize()
            label_0 = dim.split('_')[1].capitalize()
            
            prediction = label_1 if prob_class_1 >= 0.5 else label_0
            confidence = max(prob_class_1, prob_class_0) * 100
            score = prob_class_1 * 100
            
            algo = model_info.get('algorithm', 'Unknown')
            
            results[dim] = {
                'prediction': prediction,
                'confidence': confidence,
                'score': score, # % towards class 1
                'algo': algo,
                'prob_dist': [prob_class_0, prob_class_1]
            }
        except Exception as e:
            st.error(f"Prediction error for {dim}: {e}")
            
    return results, X_input

# ==========================================
# UI COMPONENTS
# ==========================================

st.title("🧠 AI Learning Style Predictor (CPL-LS)")
st.markdown("### Transparent, Explainable FSLSM Classification Panel")

if not models_dict:
    st.warning("Models are currently training or unavailable. Please wait.")
    st.stop()

# TABS
tab_predict, tab_transparency, tab_xai, tab_dataset, tab_arch = st.tabs([
    "🎯 1. Predict (Student View)", 
    "📊 2. Model Transparency", 
    "🔍 3. Explainability (SHAP)", 
    "📈 4. Dataset & Features",
    "📄 5. CPL-LS Architecture"
])

# ------------------------------------------
# TAB 1: PREDICT
# ------------------------------------------
with tab_predict:
    st.markdown("### 🎓 Student Profile Input")
    
    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**👁️ Visual-Verbal**")
            video_watch_time = st.slider('Video Time (min)', 0, 45, 25)
            diagram_view_time = st.slider('Diagram Time', 0, 30, 15)
            reading_time = st.slider('Reading Time', 0, 60, 20)
            visual_content_engagement = st.slider('Visual Content', 0, 40, 20)
            
        with col2:
            st.markdown("**💬 Active-Reflective**")
            messages_posted = st.slider('Messages Posted', 0, 50, 18)
            group_discussions = st.slider('Group Discussions', 0, 20, 8)
            hands_on_activities = st.slider('Hands-on Time', 0, 45, 20)
            
        with col3:
            st.markdown("**🔍 Sensing-Intuitive**")
            practical_exercises = st.slider('Practical Ex.', 0, 40, 22)
            detail_orientation = st.slider('Detail Focus', 0, 35, 18)
            theoretical_discussions = st.slider('Theory Discussions', 0, 35, 12)
            pattern_recognition = st.slider('Pattern Recog.', 0, 40, 15)
            
        with col4:
            st.markdown("**📚 Sequential-Global**")
            step_by_step_completion = st.slider('Step-by-Step', 0, 40, 25)
            linear_progression = st.slider('Linear Progress', 0, 35, 20)
            big_picture_focus = st.slider('Big Picture', 0, 40, 15)
            holistic_understanding = st.slider('Holistic Underst.', 0, 35, 12)
            
        submitted = st.form_submit_button("🚀 Run Live Inference", use_container_width=True)

    if submitted:
        input_data = {
            'video_watch_time': video_watch_time,
            'diagram_view_time': diagram_view_time,
            'reading_time': reading_time,
            'visual_content_engagement': visual_content_engagement,
            'messages_posted': messages_posted,
            'group_discussions': group_discussions,
            'hands_on_activities': hands_on_activities,
            'practical_exercises': practical_exercises,
            'detail_orientation': detail_orientation,
            'theoretical_discussions': theoretical_discussions,
            'pattern_recognition': pattern_recognition,
            'step_by_step_completion': step_by_step_completion,
            'linear_progression': linear_progression,
            'big_picture_focus': big_picture_focus,
            'holistic_understanding': holistic_understanding
        }
        
        st.session_state.predictions, st.session_state.user_inputs = predict_learning_style(input_data, models_dict)
        
    if st.session_state.predictions:
        st.markdown("---")
        st.markdown("### 📊 Inference Results")
        
        cols = st.columns(4)
        dimensions = [
            ('visual_verbal', '👁️ Visual vs Verbal', '#e74c3c', 'Visual', 'Verbal'),
            ('active_reflective', '💬 Active vs Reflective', '#3498db', 'Active', 'Reflective'),
            ('sensing_intuitive', '🔍 Sensing vs Intuitive', '#2ecc71', 'Sensing', 'Intuitive'),
            ('sequential_global', '📚 Sequential vs Global', '#9b59b6', 'Sequential', 'Global')
        ]
        
        for idx, (dim_key, title, color, lbl1, lbl0) in enumerate(dimensions):
            with cols[idx]:
                if dim_key in st.session_state.predictions:
                    res = st.session_state.predictions[dim_key]
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-top: 5px solid {color}; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4 style="margin-top:0; color: #2c3e50;">{title}</h4>
                        <h2 style="color: {color}; margin-bottom: 5px;">{res['prediction']}</h2>
                        <p style="font-size: 14px; margin-bottom: 0;"><b>Confidence:</b> {res['confidence']:.1f}%</p>
                        <p style="font-size: 12px; color: #7f8c8d;">Model: {res['algo']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Bar
                    fig = go.Figure(go.Indicator(
                        mode = "gauge",
                        value = res['score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{lbl0} ← → {lbl1}", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "#ecf0f1"},
                                {'range': [50, 100], 'color': "#bdc3c7"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    # --- BATCH CSV UPLOAD ---
    st.markdown("---")
    st.markdown("### 📂 Batch CSV Upload")
    st.write("Upload a CSV file with student behavioral data to run batch predictions. "
             "The CSV should contain columns matching the raw features (e.g., `T_image`, `T_video`, `T_read`, etc.).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(batch_df)}** student records with {batch_df.shape[1]} columns.")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("🚀 Run Batch Inference"):
            progress_bar = st.progress(0)
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                row_input = {
                    'video_watch_time': row.get('T_video', 0),
                    'diagram_view_time': row.get('T_image', 0),
                    'reading_time': row.get('T_read', 0),
                    'visual_content_engagement': row.get('T_powerpoint', 0),
                    'messages_posted': row.get('N_msgs_posted', 0),
                    'group_discussions': row.get('N_group_discussions', 0),
                    'hands_on_activities': row.get('T_solve_excercise', 0),
                    'practical_exercises': row.get('T_concrete', 0),
                    'detail_orientation': row.get('N_questions_on_details', 0),
                    'theoretical_discussions': row.get('T_audio', 0),
                    'pattern_recognition': row.get('N_standard_questions_correct', 0),
                    'step_by_step_completion': row.get('T_result', 0),
                    'linear_progression': row.get('N_next_button_used', 0),
                    'big_picture_focus': row.get('T_hierarchies', 0),
                    'holistic_understanding': row.get('N_questions_on_outlines', 0)
                }
                
                try:
                    preds, _ = predict_learning_style(row_input, models_dict)
                    entry = {'Student_ID': row.get('Student_ID', f'STD-{idx+1}')}
                    for dim in preds:
                        entry[f'{dim}_Prediction'] = preds[dim]['prediction']
                        entry[f'{dim}_Confidence'] = f"{preds[dim]['confidence']:.1f}%"
                        entry[f'{dim}_Model'] = preds[dim]['algo']
                    batch_results.append(entry)
                except Exception as e:
                    batch_results.append({'Student_ID': f'STD-{idx+1}', 'Error': str(e)})
                    
                progress_bar.progress((idx + 1) / len(batch_df))
            
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)
            
            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Results (CSV)",
                csv_bytes,
                "batch_predictions.csv",
                "text/csv"
            )

# ------------------------------------------
# TAB 2: MODEL TRANSPARENCY
# ------------------------------------------
with tab_transparency:
    st.markdown("### 🏆 SOTA Model Performance & Competition")
    st.write("For each dimension, 4 models competed iteratively using CPL-LS. Below are the authentic accuracies and winning algorithms chosen by the pipeline.")
    
    # 1. Real Accuracy Table
    metrics_data = []
    for dim, info in models_dict.items():
        if 'all_results' in info:
            row = {'Dimension': dim.replace('_', '-').title(), 'Winner': info['algorithm']}
            for algo, (acc, _) in info['all_results'].items():
                row[algo] = f"{acc*100:.2f}%"
            metrics_data.append(row)
            
    if metrics_data:
        st.dataframe(pd.DataFrame(metrics_data).set_index('Dimension'), use_container_width=True)
    else:
        st.info("Train the pipeline with full metadata to see the model comparison grid.")

    st.markdown("---")
    st.markdown("### 🔄 CPL-LS Threshold Evolution")
    st.write("Dynamic per-class thresholds during the semi-supervised learning iterations (warm-up phase → curriculum phase).")
    
    dim_sel = st.selectbox("Select Dimension to view CPL-LS Thresholds", list(models_dict.keys()))
    
    if dim_sel in models_dict:
        model_info = models_dict[dim_sel]
        
        # We need to extract the base model from the wrapper to check if it's our CPL model
        trained_algo = model_info['model']
        
        # Check if the model has threshold_history_ (CPL-LS models do)
        if hasattr(trained_algo, 'threshold_history_') and len(trained_algo.threshold_history_) > 0:
            history = trained_algo.threshold_history_
            
            # Plot thresholds over time
            iters = [h['iteration'] for h in history]
            classes = list(history[0]['thresholds'].keys())
            
            fig = go.Figure()
            for c in classes:
                thresh_vals = [h['thresholds'][c] for h in history]
                sigma_vals = [h['sigma'][c] for h in history]
                
                # Threshold lines
                fig.add_trace(go.Scatter(x=iters, y=thresh_vals, mode='lines+markers', 
                                         name=f'Class {c} Threshold', line=dict(dash='solid')))
                
            fig.update_layout(
                title=f"Iterative Threshold Adjustment ({dim_sel})",
                xaxis_title="Self-Training Iteration",
                yaxis_title="Confidence Threshold (τ)",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw Threshold Data
            st.write("Raw Iteration Logs:")
            history_df = pd.DataFrame([{
                'Iteration': h['iteration'],
                'Phase': h['phase'].upper(),
                **{f"Class {k} Learning Effect (σ)": v for k,v in h['sigma'].items()},
                **{f"Class {k} Threshold (T)": v for k,v in h['thresholds'].items()}
            } for h in history])
            
            st.dataframe(history_df, use_container_width=True)
            
        else:
            st.info(f"No threshold history available for the winning model ({model_info.get('algorithm')}). This usually means a non-CPL model (like TabNet-PL student) won this dimension, or exactly 0 unlabeled items were available.")

# ------------------------------------------
# TAB 3: EXPLAINABILITY (XAI)
# ------------------------------------------
with tab_xai:
    st.markdown("### 🔍 Model Explainability (SHAP & Feature Importance)")
    st.write("Total transparency into how the models are making decisions. Uses actual SHAP (SHapley Additive exPlanations) values computed on the test dataset.")
    
    xai_dim = st.selectbox("Select Dimension for XAI Analysis", list(models_dict.keys()))
    
    if xai_dim in models_dict:
        model_data = models_dict[xai_dim]
        model = model_data['model']
        algo_name = model_data.get('algorithm', '')
        
        st.write(f"**Analyzing Winner:** `{algo_name}`")
        
        # Some models don't support SHAP directly or easily (like PyTorch KAN or TabNet without wrappers)
        # So we focus XAI on tree-based winners or show feature importances
        
        if 'X_test' in model_data and 'y_test' in model_data:
            X_test = model_data['X_test']
            features = model_data.get('features', [f"Feature_{i}" for i in range(X_test.shape[1])])
            
            # If the base estimator inside CPL supports feature_importances_
            base_est = getattr(model, 'model_', model) 
            has_importances = hasattr(base_est, 'feature_importances_')
            
            if has_importances:
                importances = base_est.feature_importances_
                
                # Check for Sklearn array format vs CatBoost/XGBoost formatting
                if importances.ndim > 1: importances = np.mean(np.abs(importances), axis=0)
                
                df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
                df_imp = df_imp.sort_values('Importance', ascending=True).tail(15) # Top 15
                
                fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                             title=f"Global Feature Importance ({algo_name})",
                             color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Global feature importance not directly available for architecture: {algo_name}")
                
            # Live SHAP evaluation for Trees
            if 'Boost' in algo_name and st.button("Calculate Exact SHAP Values (Takes ~5s)"):
                with st.spinner("Computing Shapley values..."):
                    try:
                        explainer = shap.TreeExplainer(base_est)
                        # Sample 100 rows to keep it fast for Streamlit
                        shap_values = explainer.shap_values(X_test[:100])
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # For binary classification, shap_values might be a list
                        if isinstance(shap_values, list):
                            shap.summary_plot(shap_values[1], X_test[:100], feature_names=features, show=False)
                        else:
                            shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)
                            
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Could not compute SHAP for this model type: {e}")
        else:
            st.info("Test dataset metadata not found in the model file. Re-run train_semi_supervised.py with the updated metadata storage.")

# ------------------------------------------
# TAB 4: DATASET & FEATURES
# ------------------------------------------
with tab_dataset:
    st.markdown("### 📈 Real Educational Dataset Statistics")
    
    if df_raw is not None:
        st.write(f"**Loaded Dataset Shape:** `{df_raw.shape[0]} rows × {df_raw.shape[1]} columns`")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Dimensional Class Balance (Target)")
            
            # Recalculate dimensions for visualization
            for dim, vals in [('Visual', [0,1,2]), ('Active', [2,3,4]), ('Sensing', [1,3,5]), ('Sequential', [0,4,5])]:
                df_raw[dim] = df_raw['learning_style'].apply(lambda x: 1 if x in vals else 0)
                
            dims = ['Visual', 'Active', 'Sensing', 'Sequential']
            balance_data = []
            for d in dims:
                counts = df_raw[d].value_counts(normalize=True) * 100
                balance_data.append({'Dimension': d, 'Class 1 (e.g. Visual/Active)': counts.get(1,0), 'Class 0 (e.g. Verbal/Reflective)': counts.get(0,0)})
                
            fig = px.bar(pd.DataFrame(balance_data), x='Dimension', y=['Class 1 (e.g. Visual/Active)', 'Class 0 (e.g. Verbal/Reflective)'], 
                         barmode='stack', title="Initial Dataset Class Imbalance")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Feature Distributions")
            feat_sel = st.selectbox("Select feature to view distribution", [c for c in df_raw.columns if c not in ['learning_style'] + dims])
            fig = px.histogram(df_raw, x=feat_sel, marginal="box", title=f"Distribution of {feat_sel}")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### The Need for SMOTE and CPL-LS")
        st.write("""
        As visible in the balance chart above, dimensions like **Sequential/Global** and **Visual/Verbal** suffer from class imbalance. 
        1. **SMOTE** is applied to the labeled set to synthetically oversample minority classes.
        2. **Fuzzy C-Means (FCM)** adds 8 soft-clustering probability features to handle fuzzy educational behaviors.
        3. **CPL-LS** uses dynamic, per-class thresholds to prevent the model from getting stuck classifying everything into the majority class during pseudo-labeling. If we used fixed thresholds, the minority class would never pass the high confidence threshold.
        """)

# ------------------------------------------
# TAB 5: ARCHITECTURE
# ------------------------------------------
with tab_arch:
    st.markdown("### 📄 CPL-LS System Architecture")
    
    if os.path.exists(arch_img_path):
        st.image(arch_img_path, caption="End-to-End System Pipeline and MLOps Architecture")
    else:
        st.info("Architecture diagram not found at docs/architecture_diagram.png")
        
    st.markdown("---")
    st.markdown("### The Curriculum Pseudo-Labeling (CPL-LS) Algorithm")
    st.latex(r"T_t(c) = \beta_t(c) \cdot \tau_{ref}")
    st.latex(r"\beta_t(c) = \frac{\sigma_t(c)}{\max_{c'} \sigma_t(c')}")
    st.write("""
    **Where:**
    - $T_t(c)$ is the dynamic threshold for class $c$ at iteration $t$.
    - $\tau_{ref}$ is the reference confidence threshold (e.g. 0.95).
    - $\sigma_t(c)$ is the learning effect of class $c$ (fraction of unlabeled samples confidently classified as $c$).
    
    **Warm-up Phase:** If the highest learning effect $\max_{c'} \sigma_t(c')$ is below $\epsilon$ (e.g. 0.05), the model is in a warm-up phase and all thresholds are temporarily set to 0 to jumpstart learning across all classes.
    """)
