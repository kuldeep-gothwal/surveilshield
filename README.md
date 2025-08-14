# 🎯 SurveilShield

**SurveilShield** is an AI-powered CCTV application built with **Python** and **Django** to detect suspicious actions in a canteen setting — like slipping items into a pocket — in **real time** using computer vision.

---

## 🚀 Features
- Real-time CCTV feed analysis
- Detects item concealment & abnormal actions
- **YOLOv8 + MediaPipe** for object & gesture detection
- Web dashboard with live monitoring & alerts
- Incident logging with images/video
- Multi-camera support

---

## 🧱 Tech Stack
- **Backend:** Python, Django, Django REST Framework, Channels, Celery, Redis
- **AI/ML:** OpenCV, Ultralytics YOLOv8, MediaPipe, TensorFlow/PyTorch
- **Database:** MySQL / PostgreSQL
- **Alerts:** WebSocket live alerts, email/SMS (configurable)

---

## 📦 Installation
```bash
git clone https://github.com/<your-username>/surveilshield.git
cd surveilshield
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Migrate & run
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
