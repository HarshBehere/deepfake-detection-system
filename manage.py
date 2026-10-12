
from flask_migrate import Migrate


def main():
    from app import create_app, db
    import models  # Ensure database models are registered

    app = create_app()

    with app.app_context():
        db.create_all()

    print("Database tables created if they did not exist.")


if __name__ == "__main__":
    main()
