# detection/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Camera
from .services.video_processor import VideoProcessor

# Dictionary to keep track of running processors
video_processors = {}

@receiver(post_save, sender=Camera)
def camera_saved(sender, instance, created, **kwargs):
    """Start or update video processor when a camera is saved"""
    camera_id = str(instance.id)
    
    # If the camera is active, start/update the processor
    if instance.is_active:
        # Stop existing processor if running
        if camera_id in video_processors:
            video_processors[camera_id].stop()
            
        # Create new processor
        processor = VideoProcessor(
            camera_id=camera_id,
            camera_url=instance.stream_url
        )
        
        # Start the processor
        processor.start()
        
        # Store in dictionary
        video_processors[camera_id] = processor
    
    # If camera is inactive, stop processor if running
    elif camera_id in video_processors:
        video_processors[camera_id].stop()
        del video_processors[camera_id]