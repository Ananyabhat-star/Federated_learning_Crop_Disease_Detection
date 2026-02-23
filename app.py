from datetime import datetime
import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from training.inference import predict_image


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "super-secret-12345"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "federated_crop.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ------------------------------------------------
# DATABASE
# ------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(120))

# class DetectionHistory(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, nullable=False)
#     image_name = db.Column(db.String(200))
#     crop = db.Column(db.String(100))
#     disease = db.Column(db.String(100))
#     status = db.Column(db.String(50))
#     confidence = db.Column(db.Float)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class DetectionHistory(db.Model):
    __tablename__ = "detection_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    image_name = db.Column(db.String(255))
    crop = db.Column(db.String(100))
    disease = db.Column(db.String(100))
    status = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

    

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ STRONG NON-LEAF FILTER (NO ML, NO RANDOM)
def is_leaf_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # green range
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_ratio = np.sum(mask > 0) / mask.size

        return green_ratio > 0.15   # 🔴 THIS IS THE KEY FIX

    except:
        return False


# ------------------------------------------------
# ROUTES
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user_id"] = 1
        session["username"] = "User"
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        session["user_id"] = 1
        session["username"] = "User"
        return redirect(url_for("dashboard"))
    return render_template("signup.html")



@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if "user_id" not in session:
        return redirect(url_for("login"))

    crop = disease = confidence = status = None
    image_filename = None
    error = None

    if request.method == "POST":
        file = request.files.get("leaf_image")

        if not file or file.filename == "":
            error = "Please upload an image"
            return render_template("detect.html", error=error)

        if not allowed_file(file.filename):
            error = "Only JPG / PNG images are allowed"
            return render_template("detect.html", error=error)

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        image_filename = filename

        # 🔴 NON-LEAF → STOP EVERYTHING
        if not is_leaf_image(save_path):
            return render_template(
                "detect.html",
                error="❌ This is NOT a leaf image. Please upload a valid crop leaf.",
                image_filename=image_filename
            )

        # ✅ REAL MODEL PREDICTION
        result = predict_image(save_path)
        if result and "error" not in result:
            record = DetectionHistory(
        user_id=session["user_id"],
        image_name=image_filename,
        crop=result["crop"],
        disease=result["disease"],
        status=result["status"],
        confidence=result["confidence"]
        )
        db.session.add(record)
        db.session.commit()

        crop = result["crop"]
        disease = result["disease"]
        confidence = result["confidence"]
        status = result["status"]

    return render_template(
        "detect.html",
        crop=crop,
        disease=disease,
        confidence=confidence,
        status=status,
        image_filename=image_filename,
        error=error
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please login to view history.", "error")
        return redirect(url_for("login"))

    records = DetectionHistory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(DetectionHistory.timestamp.desc()).all()

    return render_template("history.html", records=records)

@app.route("/delete_history/<int:record_id>", methods=["POST"])
def delete_history(record_id):
    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))

    record = DetectionHistory.query.filter_by(
        id=record_id,
        user_id=session["user_id"]
    ).first()

    if record:
        db.session.delete(record)
        db.session.commit()
        flash("History record deleted successfully.", "success")
    else:
        flash("Record not found.", "error")

    return redirect(url_for("history"))



# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)