from app import db
from flask_login import UserMixin
import torch
import torch.nn as nn
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from sqlalchemy.orm import relationship
import traceback


# User model for database
class User(UserMixin, db.Model):
    __tablename__ = "user"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pwd = db.Column(db.String(300), nullable=False)

    results = relationship(
        "Result",
        backref="user",
        lazy=True
    )

    def __repr__(self):
        return f"<User {self.username}>"


# Result model for storing prediction results
class Result(db.Model):
    __tablename__ = "result"

    result_id = db.Column(db.Integer, primary_key=True)
    confidence_score = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    feedback = db.Column(db.String(200), nullable=True)
    image_path = db.Column(db.String(255), nullable=False)

    # Foreign key
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    def __repr__(self):
        return f"<Result {self.result_id} - Prediction: {self.prediction}>"


# Custom Vision Transformer model
class CustomViT(nn.Module):

    def __init__(self):
        super(CustomViT, self).__init__()

        model_name = "google/vit-base-patch16-224"

        # Feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            model_name
        )

        # Pretrained Vision Transformer
        self.model = ViTForImageClassification.from_pretrained(
            model_name
        )

        # Change classifier to binary classification
        self.model.classifier = nn.Sequential(
            nn.Linear(
                self.model.classifier.in_features,
                2
            ),
            nn.Dropout(0.1)
        )

        # Load trained model weights
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_ViT.pth"
            )

            path = torch.load(
                model_path,
                map_location="cpu"
            )

            self.model.load_state_dict(
                path,
                strict=False
            )

            print("Model weights loaded successfully.")

            print("Tensors and their shapes in the .pth file:")

            for key, value in path.items():
                print(f"{key}: {value.shape}")

        except Exception as e:
            print(f"Error loading model weights: {e}")

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.
        """
        outputs = self.model(x)
        return outputs.logits

    def predict(self, image):
        """
        Run inference on a PIL image.
        """

        self.eval()

        # Preprocess image
        inputs = self.feature_extractor(
            images=image,
            return_tensors="pt"
        )

        print(
            "Transformed input shape:",
            inputs["pixel_values"].shape
        )

        # Run inference without calculating gradients
        with torch.no_grad():
            outputs = self.forward(
                inputs["pixel_values"]
            )

        return outputs


# Load model
model = CustomViT()
model.eval()


# Main prediction function
def predict_image(filepath):

    try:

        # Open and process image
        image = Image.open(filepath).convert("RGB")

        print(
            f"Original image size: {image.size}"
        )

        # Get model prediction
        outputs = model.predict(image)

        print(
            f"Raw model outputs (logits): {outputs}"
        )

        # Convert logits to probabilities
        probs = torch.softmax(
            outputs,
            dim=1
        )

        print(
            f"Softmax probabilities: {probs}"
        )

        # Get confidence and predicted class
        confidence, predicted_class = torch.max(
            probs,
            dim=1
        )

        # Map class indices to labels
        label_map = {
            0: "Fake",
            1: "Real"
        }

        prediction = label_map.get(
            predicted_class.item(),
            "Unknown"
        )

        print(
            f"Prediction: {prediction}, "
            f"Confidence: {confidence.item()}"
        )

        # Prepare relative path for frontend
        filename = os.path.basename(filepath)

        image_rel_path = os.path.join(
            "uploads",
            filename
        ).replace("\\", "/")

        # Return results
        return {
            "prediction": prediction,
            "confidence": round(
                confidence.item() * 100,
                4
            ),
            "image_path": image_rel_path
        }

    except Exception as e:

        print(
            f"Error during prediction: {e}"
        )

        traceback.print_exc()

        return {
            "error": f"Error during prediction: {str(e)}"
        }