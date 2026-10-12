# Deepfake Detection System

A web-based deepfake detection project built using Python, Flask, PyTorch, and a Vision Transformer (ViT).

## Features

- User registration and login
- Password hashing and session management
- Image upload and quality validation
- Face detection and image preprocessing
- Vision Transformer-based image classification
- Prediction labels and confidence scores
- Prediction history for logged-in users
- User feedback on predictions

## Technology Stack

- Python 3.10
- Flask
- PyTorch
- Hugging Face Transformers
- OpenCV
- MediaPipe
- Pillow
- PostgreSQL
- SQLAlchemy
- Flask-Login

## Installation

### 1. Clone the repository

After creating your GitHub repository, use its URL:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

On Windows:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file using `.env.example` as a template.

Configure these variables:

```dotenv
SECRET_KEY=replace-with-a-random-secret
DATABASE_URL=postgresql://USERNAME:PASSWORD@localhost:5432/DATABASE_NAME
```

Replace the example values with your local PostgreSQL credentials. Never upload your actual `.env` file to GitHub.

### 5. Run the application

```bash
python routes.py
```

Database migration setup must be reviewed before running the application against a new database.

## Model Limitations

The project uses a Vision Transformer architecture for binary image classification. The local `best_ViT.pth` file is only 932 bytes, and its contents have not been verified as a valid trained checkpoint.

Model accuracy has not been established. Predictions and confidence scores should therefore be treated as experimental, not as reliable proof that an image is real or fake.

## Security

- Keep credentials in `.env`.
- Do not commit database files or private user uploads.
- Disable Flask debug mode before production deployment.

## License

See the LICENSE file for license details.