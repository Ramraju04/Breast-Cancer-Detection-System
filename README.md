# 🩺 Breast Cancer Detection System
### Deep Learning • Explainable AI • AI-Powered Medical Assistant

An intelligent web application designed to assist in the early detection of breast cancer from histopathological images using Deep Learning techniques. The system integrates Explainable AI through Grad-CAM visualizations and provides an AI-powered chatbot to answer breast cancer-related queries.

> ⚠️ **Disclaimer:** This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

---

## ✨ Key Highlights

- 🔬 **Breast Cancer Classification**
  - Classifies breast tissue images as **Benign** or **Malignant**.

- 📈 **Prediction Confidence**
  - Displays the model's confidence score for every prediction.

- 🔍 **Explainable AI (Grad-CAM)**
  - Generates heatmaps highlighting the regions that influenced the model's decision.

- 🤖 **AI Medical Chatbot**
  - Powered by **Groq Llama 3.1 8B Instant** to answer breast cancer awareness and treatment-related questions.

- 💊 **Treatment Awareness**
  - Provides general information about treatment approaches for malignant cases.

---

## 🖼️ System Workflow

```text
Image Upload
     ↓
Image Preprocessing
     ↓
CNN-Based Classification
     ↓
Prediction & Confidence Score
     ↓
Grad-CAM Visualization
     ↓
Result Interpretation
     ↓
AI Chatbot Assistance
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Backend | Flask, Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, NumPy |
| Explainable AI | Grad-CAM |
| Frontend | HTML, CSS |
| Conversational AI | Groq API, Llama 3.1 8B Instant |

---

## 📂 Project Structure

```text
Breast-Cancer-Detection-System/
│
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/Ramraju04/Breast-Cancer-Detection-System.git
cd Breast-Cancer-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### Add the Trained Model

Place the trained model file in the root directory:

```text
model.h5
```

> The model file is not included in this repository due to GitHub's file size limitations.

### Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🔮 Future Scope

- Multi-class breast cancer classification
- Cloud deployment
- Doctor dashboard
- Patient history management
- User authentication
- Enhanced treatment recommendation support

---

## 👨‍💻 Author

**Ramraju Bodda**

- GitHub: https://github.com/Ramraju04
- Final Year Project | Artificial Intelligence & Machine Learning

---

## 📄 License

Distributed under the **MIT License**.
