# PhoneDetect Project

Simple webcam-based phone detector using YOLOv5 via torch.hub.

Quick start (Windows PowerShell):

1. Activate virtual environment

   .\\.venv\\Scripts\\Activate.ps1

2. Install dependencies (if not already installed)

   & .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt

3. Run

   & .\\venv\\Scripts\\python.exe main.py

Notes
- Captured images are saved to the `phone_captures` directory.
- If you push this repo to GitHub, add a `requirements.txt` (e.g. `pip freeze > requirements.txt`) so others can install the same deps.