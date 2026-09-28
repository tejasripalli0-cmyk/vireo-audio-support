git clone https://github.com/tejasripalli0-cmyk/vireo-audio-support.git
cd vireo-audio-support

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python run.py

python -m pytest -q

python app.py

Start-Process "http://127.0.0.1:10000"