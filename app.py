from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    try:
        data = CustomData(
            gender=request.form["gender"],
            race_ethnicity=request.form["ethnicity"],
            parental_level_of_education=request.form["parental_level_of_education"],
            lunch=request.form["lunch"],
            test_preparation_course=request.form["test_preparation_course"],
            reading_score=float(request.form["reading_score"]),
            writing_score=float(request.form["writing_score"]),
        )
        pred_df = data.get_data_as_data_frame()
        result = PredictPipeline().predict(pred_df)[0]
        return render_template("home.html", results=f"{result:.2f}")
    except (KeyError, TypeError, ValueError):
        return render_template(
            "home.html",
            error="Please provide valid values for every field.",
        ), 400
        
if __name__ == "__main__":
    app.run(host="0.0.0.0")
