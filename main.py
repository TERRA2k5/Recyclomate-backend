from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 1),
    torch.nn.Sigmoid()
)
model = model.to(device)

model.load_state_dict(torch.load('entire_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        img = Image.open(io.BytesIO(file.read()))
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output > 0.5).float().item()

    result = 'Recyclable' if prediction == 1 else 'Organic'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
