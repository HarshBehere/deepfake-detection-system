import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# Load environment variables from the local .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Initialize Flask extensions
db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
login_manager = LoginManager()

login_manager.session_protection = "strong"
login_manager.login_view = "login"
login_manager.login_message_category = "info"


def create_app():
    app = Flask(
        __name__,
        static_url_path="/static",
        static_folder="static",
    )

    # Session configuration
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

    # Local HTTP development only.
    # Set this to True when deployed over HTTPS.
    app.config["SESSION_COOKIE_SECURE"] = False

    # Load configuration from environment variables
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    if not app.config["SECRET_KEY"]:
        raise RuntimeError(
            "SECRET_KEY is not set. Create a local .env file "
            "using .env.example as a template."
        )

    if not app.config["SQLALCHEMY_DATABASE_URI"]:
        raise RuntimeError(
            "DATABASE_URL is not set. Create a local .env file "
            "using .env.example as a template."
        )

    # Initialize extensions with the Flask application
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
