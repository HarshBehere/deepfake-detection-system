# Standard Library
import io
import base64
import os
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import matplotlib
matplotlib.use('Agg')
import traceback

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, flash, url_for, session, request, jsonify
from flask_login import UserMixin, login_user, LoginManager, current_user, logout_user, login_required
from werkzeug.utils import secure_filename
from sqlalchemy.exc import IntegrityError, DataError, DatabaseError, InterfaceError, InvalidRequestError

# Internal
from app import create_app, db, login_manager, bcrypt
from models import User, Result, predict_image
from filter_utils import filter_image
from forms import login_form,RegisterForm

# from werkzeug.routing import BuildError


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app = create_app()

@app.before_request
def session_handler():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=7)
    
def check_session():
    print("Current session:", session)

# Dashboard route
@app.route("/", methods=("GET", "POST"), strict_slashes=False)
def index():
    return render_template("index.html",title="Home")

#Login route
@app.route("/login/", methods=("GET", "POST"), strict_slashes=False)
def login():
    form = login_form()

    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()
            
            if user and bcrypt.check_password_hash(user.pwd, form.pwd.data):
                login_user(user)
                session.permanent = True
                next_page = request.args.get('next', url_for('upload'))
                return redirect(next_page)
            else:
                flash("Invalid username or password!", "danger")
        except Exception as e:
            flash(str(e), "danger")

    return render_template("auth.html",
        form=form,
        text="Login",
        title="Login",
        btn_action="Login",
    )

# Register route
@app.route("/register/", methods=("GET", "POST"), strict_slashes=False)
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        try:
            email = form.email.data
            pwd = form.pwd.data
            username = form.username.data
            hashed_password = bcrypt.generate_password_hash(pwd).decode('utf-8')  # Hash the password
            
            # Create new user
            newuser = User(
                username=username,
                email=email,
                pwd=hashed_password
            )
    
            db.session.add(newuser)
            db.session.commit()
            
            flash(f"Account Succesfully created", "success")
            return redirect(url_for("login"))

        except InvalidRequestError:
            db.session.rollback()
            flash(f"Something went wrong!", "danger")
        except IntegrityError:
            db.session.rollback()
            flash(f"User already exists! Please use a different email or username.", "warning")
        except DataError:
            db.session.rollback()
            flash(f"Invalid Entry", "warning")
        except InterfaceError:
            db.session.rollback()
            flash(f"Error connecting to the database", "danger")
        except DatabaseError:
            db.session.rollback()
            flash(f"Error connecting to the database", "danger")
        # except BuildError:
        #     db.session.rollback()
        #     flash(f"An error occured !", "danger")
    return render_template("auth.html",
        form=form,
        text="Create account",
        title="Register",
        btn_action="Register account",
        )
    
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload route
@app.route("/upload", methods=["GET", "POST"], strict_slashes=False)
@login_required
def upload():
    """
    Handle file uploads and process them through the prediction model.
    
    GET: Display the upload form
    POST: Process the uploaded file, run predictions, and redirect to results
    """
    # Ensure session is permanent before processing the request
    session.permanent = True  # Make the session last longer
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Set session expiration to 7 days
    
    if request.method == "POST":
        # Check if a file was included in the request
        if 'file' not in request.files:
            flash("No file was included in the request", "danger")
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            flash("No file was selected", "warning")
            return redirect(request.url)
        
        # Check if the file is of an allowed type
        if file and allowed_file(file.filename):
            try:
                # Secure the filename and save the file
                filename = secure_filename(file.filename)
                
                # Ensure upload directory exists
                upload_dir = os.path.join('static/uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                # Apply image quality filtering before prediction
                if not filter_image(filepath):
                    flash("Uploaded image did not meet quality standards (face detection, brightness, sharpness, etc.).", "warning")
                    os.remove(filepath)
                    return redirect(request.url)
                
                # Run prediction on the uploaded image, passing the model
                result = predict_image(filepath)
                
                # Log the raw result for debugging
                app.logger.debug(f"Raw prediction result: {result}")
                
                # Validate prediction result with more specific error handling
                if result is None:
                    flash("AI processing returned no results. The model may be unavailable.", "danger")
                    return redirect(request.url)
                if "prediction" not in result:
                    flash("AI processing was incomplete. Missing prediction data.", "danger")
                    app.logger.error(f"Incomplete prediction result: {result}")
                    return redirect(request.url)
                
                # Check for error in result
                if "error" in result:
                    flash(f"Error during prediction: {result['error']}", "danger")
                    return redirect(request.url)
                
                # Generate relative image path for storage and display
                image_path = os.path.join('uploads', filename).replace("\\", "/")
                
                # Log information for debugging
                app.logger.info(f"Prediction successful: {result['prediction']}")
                
                # Save the result to the database
                new_result = Result(
                    confidence_score=result["confidence"],
                    prediction=result["prediction"],
                    feedback=None,  # Feedback will be added later if provided
                    image_path=image_path,
                    user_id=current_user.id  # Associate with the logged-in user
                )
                
                db.session.add(new_result)
                db.session.commit()
                
                # Render the result page with prediction information
                return render_template(
                    "result.html",
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    image_path=image_path,
                    result_id=new_result.result_id
                )
                
            except Exception as e:
                # Handle any errors that occur during processing
                app.logger.error(f"Error processing upload: {str(e)}")
                app.logger.error(traceback.format_exc())  # Log the full traceback
                flash(f"An error occurred while processing your upload: {str(e)}", "danger")
                return redirect(request.url)
        else:
            flash("Unsupported file type. Please upload an allowed file format.", "danger")
            return redirect(request.url)
    
    # For GET requests, just render the upload form
    return render_template("upload.html", title="Upload Image")


#Result table route
@app.route('/my_results')
@login_required
def my_results():
    # Fetch all results related to current logged-in user
    results = Result.query.filter_by(user_id=current_user.id).all()
    return render_template('my_results.html', results=results)

@app.route('/result/<int:result_id>')
@login_required
def result_page(result_id):
    result = Result.query.filter_by(
        result_id=result_id,
        user_id=current_user.id
    ).first()

    if not result:
        flash("Result not found", "danger")
        return redirect(url_for("my_results"))

    return render_template(
        "result.html",
        prediction=result.prediction,
        confidence=result.confidence_score,
        image_path=result.image_path,
        result_id=result.result_id
    )
@app.route('/submit_feedback', methods=['POST'], strict_slashes=False)
@login_required
def submit_feedback():
    result_id = request.form['result_id']
    feedback = request.form['feedback']  # "Satisfied" or "Unsatisfied"
    
    if not result_id or not result_id.isdigit():
        flash("Invalid result ID!", "danger")
        return redirect(url_for('my_results'))
    
    result = Result.query.filter_by(
    result_id=result_id,
    user_id=current_user.id
    ).first()
    
    if not result:
        flash("Result not found!", "danger")
        return redirect(url_for('my_results'))

    # Apply feedback logic
    if feedback == "Satisfied":
        result.feedback = result.prediction  # Feedback same as prediction
    elif feedback == "Unsatisfied":
        # Flip the feedback if Unsatisfied
        result.feedback = "Fake" if result.prediction == "Real" else "Real"  # Flipped feedback
    else:
        flash("Invalid feedback option!", "danger")
        return redirect(url_for('my_results'))

    try:
        db.session.commit()
        flash("Thank you for your feedback!", "success")
    except IntegrityError:
        db.session.rollback()
        flash("Database integrity error while saving feedback. Please try again.", "danger")
    except DatabaseError:
        db.session.rollback()
        flash("An error occurred with the database. Please try again later.", "danger")
    except Exception as e:
        db.session.rollback()
        flash("An unknown error occurred. Please try again.", "danger")
        print("Commit failed:", str(e))

    return redirect(url_for('upload'))  # Redirect to the upload screen after feedback submission


# Function to generate the radar chart
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(y_true, y_pred):
    """
    Generate a confusion matrix plot and save it as an image.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save the confusion matrix plot to a static file
    cm_image_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(cm_image_path)
    plt.close()  # Close the plot to avoid overlapping with future plots

    return cm_image_path


#logout table rooute
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)