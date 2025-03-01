import os
from celery import Celery
print('hiihhh')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'surveilshield.settings')
print('hii')
app = Celery('surveilshield')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()