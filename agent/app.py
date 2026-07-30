import sys
import os
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# ==========================================
# DYNAMIC PATH RESOLUTION
# ==========================================
# Ensure current directory and parent directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

for folder in [CURRENT_DIR, PARENT_DIR, os.path.join(CURRENT_DIR, 'map'), os.path.join(PARENT_DIR, 'map')]:
    if os.path.exists(folder) and folder not in sys.path:
        sys.path.append(folder)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(CURRENT_DIR, ".env"))

# Safe Imports
from agent_runner import run_agent

# Import map_assistant with fallback if placed in subfolder
try:
    from map_assistant import answer_map_question
except ModuleNotFoundError:
    try:
        from map.map_assistant import answer_map_question
    except ModuleNotFoundError:
        # Fallback dummy function if map_assistant.py is missing/renamed
        def answer_map_question(msg: str):
            return {
                "answer": "Map assistant module could not be loaded. Please check file placement.",
                "source": "error"
            }

# ==========================================
# FLASK SERVER INITIALIZATION
# ==========================================
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    dashboard_path = os.path.join(CURRENT_DIR, "nc_internet_vs_eog_dashboard.html")
    generate_script = os.path.join(CURRENT_DIR, "generate_dashboard.py")
    
    if os.path.exists(generate_script):
        subprocess.run(["python", generate_script])
        
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    return "Dashboard generation script running or file missing.", 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "")
    persona = data.get("persona", "general")
    history = data.get("history", [])

    try:
        response = run_agent(
            message=message,
            persona=persona,
            history=history
        )

        return jsonify({
            "response": response
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "response": f"Agent error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)