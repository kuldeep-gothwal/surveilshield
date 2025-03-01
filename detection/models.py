# detection/models.py
from django.db import models
from django.utils import timezone
import uuid

class Camera(models.Model):
    """
    Model representing a CCTV camera in the system.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    stream_url = models.URLField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Camera settings
    resolution = models.CharField(max_length=20, default="1280x720")
    fps = models.IntegerField(default=30)

    def __str__(self):
        return f"{self.name} ({self.location})"


class DetectionSettings(models.Model):
    """
    Settings for the detection system.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Detection parameters
    confidence_threshold = models.FloatField(default=0.7)
    motion_threshold = models.IntegerField(default=500)
    frame_skip = models.IntegerField(default=5)
    alert_cooldown = models.IntegerField(default=30)  # seconds

    # Notification settings
    send_email = models.BooleanField(default=True)
    send_sms = models.BooleanField(default=False)
    use_webhook = models.BooleanField(default=False)
    webhook_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({'Active' if self.is_active else 'Inactive'})"


class Alert(models.Model):
    """
    Model for storing theft-detection alerts.
    """
    STATUS_CHOICES = [
        ('new', 'New'),
        ('reviewing', 'Under Review'),
        ('confirmed', 'Confirmed Theft'),
        ('false_alarm', 'False Alarm'),
        ('resolved', 'Resolved')
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='alerts')
    timestamp = models.DateTimeField(default=timezone.now)
    detection_data = models.JSONField(default=dict)
    video_evidence = models.FileField(upload_to='alerts/', blank=True, null=True)
    image_snapshot = models.ImageField(upload_to='snapshots/', blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Alert {self.id} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def is_new(self):
        return self.status == 'new'

    @property
    def is_theft_confirmed(self):
        return self.status == 'confirmed'