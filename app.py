import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from data_loader import load_data,CLASS_NAMES,plot_sample

st.set_page_config(page_title='ANN vs ML-- Fashion MNIST',layout='wide',page_icon='🧠')


#Load The data

@st.cache_data
def load_all():
    (X_train,y_train),(X_test,y_test)=load_data()
    X_test_norm = X_test.astype('float32') / 255.0

    with open('results/ml_results.json') as f:
        results=json.load(f)

    with open('results/ann_history.json') as f:
        history=json.load(f)
    
    preds = np.load('results/y_pred_all.npy',allow_pickle=True).item()

    return X_train, y_train, X_test, y_test, X_test_norm, results, history, preds

X_train, y_train, X_test, y_test, X_test_norm, results, history, preds = load_all()

# Tabs

tabs = st.tabs([
    "🎯 Introduction",
    "📊 Dataset Overview",
    "🤖 ML Model Results",
    "🧠 ANN Explanation",
    "📈 Model Comparison",
    "🔮 Try Prediction",
    "💡 Key Insights"
    ])

# Tab 1: Introduction
with tabs[0]:
    st.header("Why Do We Need Neural Networks?")
    st.markdown("""
    Machine Learning models like **Logistic Regression**, **SVM**, and **Random Forest**
    are powerful — but they have limitations when it comes to learning from raw data.

    This project demonstrates those limitations by comparing them against
    an **Artificial Neural Network** on the **Fashion MNIST** dataset.

    ---

    ### 🎯 Central Question
    > *What does ANN add that traditional ML models cannot?*

    ### 📋 Models Compared
    | Type | Models |
    |---|---|
    | **Traditional ML** | Logistic Regression, SVM, Random Forest |
    | **Deep Learning** | Artificial Neural Network (ANN) |
    """)

# Tab 2: Dataset Overview
with tabs[1]:
    st.header("Fashion MNIST Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", "60,000")
        st.metric("Test Samples", "10,000")
        st.metric("Image Size", "28 × 28 px")
        st.metric("Classes", "10")
        st.markdown("---")
        st.subheader("Class Distribution")
        unique, counts = np.unique(y_train, return_counts=True)
        dist_df = pd.DataFrame({'Class': [CLASS_NAMES[i] for i in unique], 'Count': counts})
        st.bar_chart(dist_df.set_index('Class'))
    with col2:
        st.subheader("Sample Images (one per class)")
        fig = plot_sample(X_train, y_train)
        st.pyplot(fig)

# Tab 3: ML Model Results
with tabs[2]:
    st.header("Traditional ML Results")
    selected = st.selectbox("Select a Model", ["Logistic Regression", "SVM", "Random Forest"])
    res = results[selected]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
        st.metric("Training Time", f"{res['train_time']:.1f}s")
        st.markdown("---")
        st.subheader("Classification Report")
        st.text(res['report'])
    with col2:
        st.subheader("Confusion Matrix")
        fname = selected.lower().replace(" ", "_")
        st.image(f"results/plots/{fname}_cm.png", use_container_width=True)

# Tab 4: ANN Explanation
with tabs[3]:
    st.header("How the ANN Learns")
    st.markdown("""
    Unlike ML models that work with **raw pixel vectors**, the ANN learns
    **internal representations** through its layers:

    | Layer | Neurons | What it learns |
    |---|---|---|
    | Input | 784 (28×28 flattened) | Raw pixel values |
    | Hidden 1 | 256 | Basic edges and patterns |
    | Hidden 2 | 128 | Combines edges into shapes |
    | Hidden 3 | 64 | Recognizes garment parts |
    | Output | 10 | Final classification |
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Curves")
        st.image("results/plots/ann_training_curves.png", use_container_width=True)
    with col2:
        st.subheader("Observations")
        st.markdown(f"""
        - **Final Accuracy:** {results['ANN']['accuracy']*100:.2f}%
        - **Training Time:** {results['ANN']['train_time']:.1f}s
        - Loss drops steeply in first 5 epochs → **fast initial learning**
        - Train/Val curves stay close → **no severe overfitting**
        """)
    st.subheader("ANN Confusion Matrix")
    st.image("results/plots/ann_cm.png", use_container_width=True)

# Tab 5: Model Comparison
with tabs[4]:
    st.header("ML vs ANN — Side by Side")
    comp_data = []
    for name, res in results.items():
        comp_data.append({
            'Model': name,
            'Accuracy (%)': round(res['accuracy'] * 100, 2),
            'Training Time (s)': round(res['train_time'], 2),
            'Learns Features': 'Yes' if name == 'ANN' else 'No',
            'Type': 'Deep Learning' if name == 'ANN' else 'Traditional ML'
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        st.image("results/plots/accuracy_comparison.png", use_container_width=True)
    with col2:
        st.subheader("Key Differences")
        st.markdown("""
        | Dimension | ML Models | ANN |
        |---|---|---|
        | Feature Learning | ❌ Manual | ✅ Automatic |
        | Accuracy | 84-89% | ~90% |
        | Scalability | Limited | Scales well |
        | Training Speed | Fast (except SVM) | Moderate |
        """)

# Tab 6: Try Prediction
with tabs[5]:
    st.header("See Models in Action")
    idx = st.slider("Pick a test sample index", 0, len(X_test) - 1, 0)
    img = X_test[idx]
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(img, width=200, caption=f"Actual: {CLASS_NAMES[y_test[idx]]}")
    with col2:
        st.subheader("Model Predictions")
        for name, y_pred in preds.items():
            pred_label = CLASS_NAMES[y_pred[idx]]
            icon = "✅" if y_pred[idx] == y_test[idx] else "❌"
            st.write(f"**{name}:** {pred_label} {icon}")

# Tab 7: Key Insights
with tabs[6]:
    st.header("What We Learned")
    st.markdown("""
    ### 🔑 Key Takeaways
    1. **ML models are not obsolete** — they achieve 84-89% accuracy with minimal setup
    2. **ANNs learn features automatically** — no manual feature engineering needed
    3. **The accuracy gap is modest (~2-5%)** on this dataset
    4. **ANN shines when data is complex** — images, text, audio
    5. **No universal winner** — depends on data complexity and needs

    ---

    ### 💡 When to Use What?
    | Scenario | Best Choice |
    |---|---|
    | Small tabular data | ML (Random Forest, SVM) |
    | Image / text / audio | ANN / Deep Learning |
    | Need interpretability | ML (Logistic Regression) |
    | Large complex dataset | ANN |
    """)
