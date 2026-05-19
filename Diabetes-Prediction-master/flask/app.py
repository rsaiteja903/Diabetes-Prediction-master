import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session
import pickle

app = Flask(__name__)

app.secret_key = "super_secret_key_change_me"

model = pickle.load(open('model.pkl', 'rb'))


def get_recent_history(limit=5):
    """Return the last N predictions from session."""
    history = session.get("history", [])
    return history[-limit:]


@app.route('/')
def home():
    return render_template(
        'index.html',
        active_tab="single",
        history=get_recent_history()
    )


# ---------- SINGLE PREDICTION ----------
@app.route('/predict', methods=['POST'])
def predict():
    fields = ["glucose", "insulin", "bmi", "age"]
    float_features = []
    for f in fields:
        val = request.form.get(f)
        float_features.append(float(val))

    final_features = np.array([float_features])

    prediction = model.predict(final_features)[0] 

    if prediction == 1:
        pred_text = "You have Diabetes, please consult a Doctor."
    else:
        pred_text = "You don't have Diabetes."

    record = {
        "Glucose": float_features[0],
        "Insulin": float_features[1],
        "BMI": float_features[2],
        "Age": float_features[3],
        "Prediction_Text": pred_text
    }
    history = session.get("history", [])
    history.append(record)
    history = history[-10:]
    session["history"] = history

    return render_template(
        'index.html',
        prediction_text=pred_text,
        active_tab="single",
        history=get_recent_history()
    )


# ---------- BATCH UPLOAD ----------
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Handle CSV upload and run predictions on all rows.
    CSV must have columns: Glucose, Insulin, BMI, Age
    """
    file = request.files.get('file')
    if file is None or file.filename == "":
        error = "Please choose a CSV file to upload."
        return render_template(
            'index.html',
            batch_error=error,
            active_tab="batch",
            history=get_recent_history()
        )

    try:
        df = pd.read_csv(file)

        required_cols = ["Glucose", "Insulin", "BMI", "Age"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            error = f"Missing columns in CSV: {', '.join(missing)}"
            return render_template(
                'index.html',
                batch_error=error,
                active_tab="batch",
                history=get_recent_history()
            )

        X = df[required_cols].values
        preds = model.predict(X)

        df["Prediction"] = preds
        df["Prediction_Text"] = df["Prediction"].map({
            0: "No Diabetes",
            1: "You have Diabetes, please consult a Doctor"
        })

        table_html = df.to_html(classes="table table-striped",
                                index=False)

        csv_data = df.to_csv(index=False)

        return render_template(
            'index.html',
            batch_table=table_html,
            batch_csv=csv_data,
            active_tab="batch",
            history=get_recent_history()
        )
    except Exception as e:
        error = f"Error processing file: {e}"
        return render_template(
            'index.html',
            batch_error=error,
            active_tab="batch",
            history=get_recent_history()
        )


if __name__ == "__main__":
    app.run(debug=True)
