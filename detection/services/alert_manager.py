# detection/services/alert_manager.py
import logging
import json
from datetime import datetime
import os
from django.conf import settings
from django.core.mail import send_mail
from django.urls import reverse
from detection.models import Alert, Camera

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.notification_methods = self.settings.get('notification_methods', ['dashboard'])
        self.alert_recipients = self.settings.get('alert_recipients', [])
        
        logger.info("Alert manager initialized")
    
    def create_alert(self, alert_data):
        """
        Create a new alert and notify recipients.
        """
        camera_id = alert_data.get('camera_id')
        timestamp = alert_data.get('timestamp', datetime.now())
        detection_results = alert_data.get('detection_results', {})
        video_evidence = alert_data.get('video_evidence')
        
        try:
            # Fetch the camera object. Alternatively, store just ID if not found.
            camera_obj = None
            try:
                camera_obj = Camera.objects.get(id=camera_id)
            except Camera.DoesNotExist:
                logger.warning(f"No Camera found for ID: {camera_id}")
            
            alert = Alert.objects.create(
                camera=camera_obj,
                timestamp=timestamp,
                detection_data=json.dumps(detection_results),
                video_evidence=video_evidence,
                status='new'
            )
            
            # Send notifications based on configured methods
            self._send_notifications(alert)
            
            logger.info(f"Created alert ID {alert.id} for camera {camera_id}")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
            return None
    
    def _send_notifications(self, alert):
        """
        Send notifications through configured channels.
        """
        for method in self.notification_methods:
            try:
                if method == 'email':
                    self._send_email_notification(alert)
                elif method == 'sms':
                    self._send_sms_notification(alert)
                elif method == 'webhook':
                    self._send_webhook_notification(alert)
                # 'dashboard' implies just storing it in DB so it shows up in UI.
            except Exception as e:
                logger.error(f"Failed to send {method} notification: {str(e)}")
    
    def _send_email_notification(self, alert):
        """
        Send email notification about the alert.
        """
        try:
            if not self.alert_recipients:
                logger.warning("No email recipients configured for alerts")
                return
            
            alert_url = f"{settings.SITE_URL}{reverse('dashboard:alert_detail', args=[alert.id])}"
            
            subject = f"CanteenGuard: Possible Theft Detected - Alert #{alert.id}"
            message = (
                f"A potential theft has been detected.\n\n"
                f"Alert ID: {alert.id}\n"
                f"Camera: {alert.camera.name if alert.camera else 'Unknown'}\n"
                f"Time: {alert.timestamp}\n\n"
                f"Review this alert: {alert_url}\n\n"
                f"Stay vigilant. (Automated system message)"
            )
            
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                self.alert_recipients,
                fail_silently=False,
            )
            
            logger.info(f"Sent email notification for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
    
    def _send_sms_notification(self, alert):
        """
        Placeholder for SMS notification integration.
        """
        logger.info(f"SMS notification for alert {alert.id} (not implemented)")
    
    def _send_webhook_notification(self, alert):
        """
        Placeholder for webhook notification integration.
        """
        logger.info(f"Webhook notification for alert {alert.id} (not implemented)")
    
    def update_alert_status(self, alert_id, new_status, notes=None):
        """
        Update the status of an existing alert.
        """
        try:
            alert = Alert.objects.get(id=alert_id)
            alert.status = new_status
            if notes:
                alert.notes = notes
            alert.last_updated = datetime.now()
            alert.save()
            
            logger.info(f"Updated alert {alert_id} status to {new_status}")
            return True
            
        except Alert.DoesNotExist:
            logger.error(f"Alert {alert_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to update alert status: {str(e)}")
            return False
    
    def get_recent_alerts(self, limit=10, status=None):
        """
        Get recent alerts, optionally filtered by status.
        """
        query = Alert.objects.all().order_by('-timestamp')
        if status:
            query = query.filter(status=status)
        return query[:limit]