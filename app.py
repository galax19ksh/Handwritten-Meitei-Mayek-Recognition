from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from transforms import transform  # Assuming your transformations are defined in a separate file named 'transforms.py'
from mmr_model import CNNModel  # Assuming your model is defined in a file named 'mmr_model.py'

# Load the model (assuming it's saved as 'mmr_model2.pt')
model = torch.load('mmr_model2.pt', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

app = Flask(__name__)

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, prediction, and returns the predicted character as JSON.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Preprocess the image
        image = Image.open(file)
        #transform = transforms.Compose([
        #    transforms.Resize((64, 64)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #])
        #image = transform(image)
        #image = image.unsqueeze(0)  # Add a batch dimension
        
        image = image.convert('RGB')  # Convert to RGB if the image is grayscale
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
 

        # Move the input tensor to the device (if GPU is available)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        # Make a prediction
        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax(dim=1).item()

        # Get the class label based on the predicted class index (assuming you have a list of class labels)
        class_labels = ["ama","ani","ahum","mari","manga","taruk","taret","nipal","mapal","phun","kok","sam","lai","mit","paa","naa","chil","til","khou","ngou","thou","wai","yang","huk","un","ee","pham","atiya","gok","jham","rai","baa","jil","dil","ghou","dhou","bham","kok lonsum","lai lonsum","mit lonsum","pa lonsum","na lonsum","til lonsum","ngou lonsum","ee lonsum","aatap","yetnap","unap","enap","cheinap","otnap","sounap","nung","cheikhei","apun", ]  # Replace with your actual class labels
        predicted_label = class_labels[predicted_class]

        return jsonify({'character': predicted_label})

    return jsonify({'error': 'Unsupported file format'}), 415

if __name__ == '__main__':
    app.run(debug=True)
