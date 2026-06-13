from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import traceback
from tensorflow.keras.models import load_model
import tensorflow as tf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('model.h5')

# Warm-up
model.predict(np.zeros((1,224,224,3)), verbose=0)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

DISCLAIMER = "⚠️ This system is for assistance only. Always consult a certified oncologist."

DRUG_RECOMMENDATIONS = {
    "Chemotherapy": "Destroys cancer cells",
    "Hormone Therapy": "Blocks cancer hormones",
    "Targeted Therapy": "Targets cancer cells"
}

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 🔥 FIXED GRADCAM
def generate_gradcam(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found")

        img_resized = cv2.resize(img, (224, 224))
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        img_tensor = tf.convert_to_tensor(img_array)

        # 🔥 get last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise RuntimeError("No Conv2D layer found")

        # 🔥 FORCE graph connection using call()
        def gradcam_model(x):
            conv_output = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv_layer.name:
                    conv_output = x
            return conv_output, x  # x = final output

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_tensor, tf.float32)
            tape.watch(inputs)

            conv_outputs, predictions = gradcam_model(inputs)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("❌ STILL None → forcing alternative gradient")
            grads = tape.gradient(tf.reduce_mean(predictions), conv_outputs)

        if grads is None:
            print("❌ FINAL FAIL → returning original")
            return img_path

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(img, 0.3, heatmap, 0.9, 0)

        base, ext = os.path.splitext(img_path)
        gradcam_path = base + "_gradcam" + ext
        cv2.imwrite(gradcam_path, superimposed)

        return gradcam_path

    except Exception as e:
        print("GradCAM ERROR:", e)
        traceback.print_exc()
        return img_path

def get_explanation(prediction, confidence):
    if prediction == 'Malignant':
        return f"Detected abnormal cells with {confidence:.1f}% confidence."
    else:
        return f"Normal tissue detected with {confidence:.1f}% confidence."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        filename = image.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(path)

        img = preprocess_image(path)
        pred = model.predict(img, verbose=0)

        confidence = float(pred[0][0]) * 100
        prediction = 'Malignant' if pred[0][0] > 0.5 else 'Benign'

        if prediction == 'Benign':
            confidence = 100 - confidence

        gradcam_path = generate_gradcam(path, model)

        response = {
            'prediction': prediction,
            'confidence': round(confidence,2),
            'image_path': '/' + path.replace("\\","/"),
            'gradcam_path': '/' + gradcam_path.replace("\\","/"),
            'explanation': get_explanation(prediction, confidence),
            'disclaimer': DISCLAIMER
        }

        if prediction == 'Malignant':
            response['drug_recommendations'] = DRUG_RECOMMENDATIONS

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_message = data.get("message","")

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"Answer only breast cancer queries simply."},
                {"role":"user","content":user_message}
            ]
        )

        return jsonify({'response': completion.choices[0].message.content})

    except Exception as e:
        return jsonify({'response': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)