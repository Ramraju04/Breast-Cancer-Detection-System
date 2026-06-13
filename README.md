# 🩺 Breast Cancer Detection System using Deep Learning and Explainable AI

A web-based intelligent system for detecting breast cancer from histopathological images using Deep Learning, Explainable AI (Grad-CAM), and an AI-powered medical chatbot.

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection significantly improves treatment outcomes.

This project provides an end-to-end solution that:

- Detects whether a breast tissue image is **Benign** or **Malignant**.
- Displays prediction confidence scores.
- Generates **Grad-CAM visualizations** to explain model decisions.
- Provides simple explanations of the prediction.
- Offers an AI-powered chatbot for answering breast cancer-related queries.

---

## 🚀 Features

### 🔬 Breast Cancer Classification
- Upload histopathological breast tissue images.
- Predicts:
  - **Benign**
  - **Malignant**

### 📊 Confidence Score
- Displays prediction confidence percentage.

### 🔍 Explainable AI (Grad-CAM)
- Highlights image regions influencing the model's prediction.
- Improves transparency and interpretability.

### 💊 Drug Recommendation Support
For malignant cases, the system provides general treatment information such as:

- Chemotherapy
- Hormone Therapy
- Targeted Therapy

> ⚠️ These are informational suggestions only and not medical prescriptions.

### 🤖 AI Medical Chatbot
Powered by **Groq Llama 3.1 8B Instant**.

Users can ask questions related to:
- Breast cancer symptoms
- Prevention methods
- Treatment options
- General awareness

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### Machine Learning & Deep Learning
- TensorFlow / Keras
- NumPy
- OpenCV

### Frontend
- HTML
- CSS

### Explainable AI
- Grad-CAM (Gradient-weighted Class Activation Mapping)

### AI Chatbot
- Groq API
- Llama 3.1 8B Instant

---

## 📂 Project Structure

```
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

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ramraju04/Breast-Cancer-Detection-System.git
cd Breast-Cancer-Detection-System
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

---

### 5. Add the Trained Model

Place the trained model file:

```
model.h5
```

in the project root directory.

> Note: The model file is not included in this repository because it exceeds GitHub's file size limitations.

---

### 6. Run the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🔄 Workflow

1. User uploads a histopathological image.
2. Image preprocessing is performed.
3. The CNN model predicts:
   - Benign
   - Malignant
4. Confidence score is calculated.
5. Grad-CAM heatmap is generated.
6. Results are displayed.
7. Users can interact with the chatbot for additional information.

---

## 📸 Explainable AI

Grad-CAM helps visualize which regions of the image influenced the model's decision.

Benefits:
- Improves trust in predictions.
- Enhances interpretability.
- Supports medical understanding.

---

## ⚠️ Disclaimer

This system is developed for **educational and research purposes only**.

It is **not intended to replace professional medical diagnosis or treatment**.

Always consult qualified healthcare professionals for medical advice.

---

## 🔮 Future Enhancements

- Multi-class breast cancer classification.
- User authentication.
- Patient history management.
- Cloud deployment.
- Doctor dashboard.
- Enhanced treatment recommendation engine.

---

## 👨‍💻 Author

**Ramraju Bodda**

GitHub: https://github.com/Ramraju04

---

## 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
