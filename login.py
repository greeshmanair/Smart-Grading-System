from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Load student credentials from Excel
def load_credentials():
    file_path = "student_performance_dataset_large.xlsx"
    df = pd.read_excel(file_path, usecols=["Register Number", "Password"])
    return df

# Authentication function
def authenticate(register_number, password):
    credentials_df = load_credentials()
    user = credentials_df[
        (credentials_df["Register Number"] == register_number) & 
        (credentials_df["Password"] == password)
    ]
    return not user.empty

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        register_number = request.form["register_number"]
        password = request.form["password"]

        if authenticate(register_number, password):
            return redirect(url_for("grad"))  # Redirect to grad.py
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/grad")
def grad():
    return "Welcome to Grad Page! Login successful."

if __name__ == "__main__":
    app.run(debug=True)
