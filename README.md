<div align="center">

# 🧠 ANN vs Traditional ML

### *Why do we need Neural Networks if ML algorithms already exist?*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

An interactive project comparing **Artificial Neural Networks** against **traditional Machine Learning** models on Fashion MNIST — demonstrating *what* ANN adds and *why* it matters.

</div>

---

## 🎯 The Question

> Traditional ML models like Logistic Regression, SVM, and Random Forest are powerful.
> So **why do we need ANNs?**

This project answers that by training both ML and ANN models on the **same dataset**, comparing them across multiple dimensions, and presenting the results through an **interactive Streamlit dashboard**.

---

## 📊 Models Compared

| Type | Model | Accuracy | Training Time |
|:---:|:---:|:---:|:---:|
| 🤖 ML | Logistic Regression | ~84% | ~30s |
| 🤖 ML | SVM (RBF Kernel) | ~88% | ~10 min |
| 🤖 ML | Random Forest | ~87% | ~1 min |
| 🧠 DL | **ANN (3 Hidden Layers)** | **~90%** | ~1 min |

---

## 🏗️ Project Structure

```
ANN/
├── 📄 data_loader.py      → Load & prepare Fashion MNIST
├── 📄 ml_models.py        → Train Logistic Regression, SVM, Random Forest
├── 📄 ann_model.py        → Build & train ANN with Keras
├── 📄 compare.py          → Generate comparison charts & plots
├── 📄 run_pipeline.py     → Orchestrate full training pipeline
├── 📄 app.py              → Streamlit interactive dashboard
├── 📄 visualizer.py       → ANN architecture diagram
└── 📁 results/
    ├── ml_results.json    → Saved model metrics
    ├── ann_history.json   → ANN training history
    └── 📁 plots/          → All generated visualizations
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Ryanrezzz/ANN.git
cd ANN
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow streamlit
```

### 2. Train All Models

```bash
python run_pipeline.py
```

> ⏱️ This takes ~15 mins (SVM is slow on 60k samples). Results are saved to `results/`.

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Preview

The Streamlit app has **7 interactive sections**:

| Tab | What It Shows |
|:---|:---|
| 🎯 **Introduction** | Project narrative and central question |
| 📊 **Dataset Overview** | Fashion MNIST samples, class distribution |
| 🤖 **ML Model Results** | Accuracy, confusion matrix, classification report per model |
| 🧠 **ANN Explanation** | Layer-by-layer learning, training curves |
| 📈 **Model Comparison** | Side-by-side accuracy, speed, feature learning |
| 🔮 **Try Prediction** | Pick any test image → see all models predict |
| 💡 **Key Insights** | When to use ML vs ANN |

---

## 🧠 ANN Architecture

```
Input (28×28) → Flatten (784)
    → Dense(256, ReLU) → Dropout(0.3)
    → Dense(128, ReLU) → Dropout(0.2)
    → Dense(64, ReLU)
    → Dense(10, Softmax) → Output
```

---

## 🔑 Key Insights

| Dimension | Traditional ML | ANN |
|:---|:---:|:---:|
| Feature Learning | ❌ Manual (raw pixels) | ✅ Automatic |
| Accuracy | 84–89% | ~90% |
| Scalability | Limited | Scales well |
| Training Speed | Fast (except SVM) | Moderate |
| Best For | Tabular data | Images, text, audio |

### The Big Takeaway

> ML models treat each pixel **independently**.
> ANNs learn **relationships between pixels** through hidden layers — that's why they perform better on visual data.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — ANN model
- **scikit-learn** — ML baselines
- **Streamlit** — Interactive dashboard
- **Matplotlib & Seaborn** — Visualizations
- **NumPy & Pandas** — Data handling

---

<div align="center">

### Built with ❤️ by [Ryan Maroof](https://github.com/Ryanrezzz)

*This project is part of my deep learning journey — understanding what makes neural networks special.*

</div>
