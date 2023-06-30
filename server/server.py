import os
import json
from flask import Flask, request, session
from werkzeug.utils import secure_filename
from constants import EXISTING_MODELS, THRESHOLDS
from ml.ml import AnomalyDetector

UPLOAD_FOLDER = os.path.abspath('./uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = "key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload():
    if not len(request.files):
        return json.dumps({
            "status": "Failure",
            "message": "No file uploaded"
        })

    # create a detector from seleted model
    model = session.get("model", "centerloss")
    model_filename = f"export_{model}.pkl"
    train_emb_path=os.path.abspath(f'ml/models/train_embs_{model}.pt')
    threshold = THRESHOLDS[model]
    detector = AnomalyDetector(
        model_filename=model_filename,
        train_emb_path=train_emb_path,
        threshold=threshold,
        data_path=app.config['UPLOAD_FOLDER'])

    # get predictions
    filepath_lst = []
    for file in request.files.values():
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath_lst.append(filepath)
    preds = detector.get_prediction(filepath_lst)
    preds = [int(i) for i in preds]
    final_ret = {
        "status": "Success",
        "prediction": list(preds)
    }

    # remove uploaded files after predictions are extracted
    for fp in filepath_lst:
        os.unlink(fp)

    return json.dumps(final_ret)

@app.route("/model", methods=["POST"])
def selectModel():
    model = request.get_data().decode('ASCII')
    if model not in EXISTING_MODELS:
        return json.dumps({
            "status": "Failure",
            "message": "Selected model does not exist"
        })
    session["model"] = model
    return json.dumps({
        "status": "Success"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')