from flask import Flask, render_template, request
import os
from predict import predict_digit

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    # Allowed file extensions
    allowed = {"png", "jpg", "jpeg"}

    if "." not in file.filename:
        return render_template(
            "index.html",
            error="Please upload a valid image."
        )

    extension = file.filename.rsplit(".", 1)[1].lower()

    if extension not in allowed:
        return render_template(
            "index.html",
            error="Only PNG, JPG and JPEG files are allowed."
        )

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(filepath)

    digit, confidence, top3 = predict_digit(filepath)

    return render_template(
        "index.html",
        prediction=digit,
        confidence=round(confidence, 2),
        image=file.filename,
        top3=top3
    )


if __name__ == "__main__":
    app.run(debug=True)