from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from agent_runner import run_agent
import subprocess

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    subprocess.run(["python", "generate_dashboard.py"])
    return send_file("nc_internet_vs_eog_dashboard.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    persona = data.get("persona", "general")
    history = data.get("history", [])
    try:
        response = run_agent(message=message, persona=persona, history=history)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Agent error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)