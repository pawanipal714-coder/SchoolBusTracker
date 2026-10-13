
from flask import Flask, render_template, request

app = Flask(__name__)

students = [
    {"id": 1, "name": "Aarav Sharma", "class": "5-A",
     "bus": "BUS-01", "status": "On Board"},
    {"id": 2, "name": "Ananya Verma", "class": "6-B",
     "bus": "BUS-01", "status": "On Board"},
    {"id": 3, "name": "Reyansh Singh", "class": "4-A",
     "bus": "BUS-02", "status": "Not Boarded"}
]

@app.route("/")
def home():
    search = request.args.get("search", "").strip().lower()
    filtered = [
        s for s in students
        if search in s["name"].lower()
        or search in s["bus"].lower()
        or search in s["class"].lower()
    ]
    return render_template("index.html", students=filtered,
                           search=search)

@app.route("/update/<int:student_id>", methods=["POST"])
def update_status(student_id):
    status = request.form.get("status")
    allowed = ["On Board", "Not Boarded", "Dropped Off"]

    for student in students:
        if student["id"] == student_id and status in allowed:
            student["status"] = status
            break

    return render_template("index.html", students=students,
                           search="")

if __name__ == "__main__":
    app.run(debug=True)
