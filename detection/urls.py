# # detection/urls.py
# from django.urls import path
# from . import views

# app_name = 'detection'

# urlpatterns = [
#     path('cameras/', views.CameraListAPIView.as_view(), name='camera_list'),
#     path('cameras/<uuid:camera_id>/', views.CameraDetailAPIView.as_view(), name='camera_detail'),
#     path('cameras/<uuid:camera_id>/feed/', views.CameraFeedView.as_view(), name='camera_feed'),
#     path('alerts/', views.AlertListAPIView.as_view(), name='alert_list'),
#     path('alerts/<uuid:alert_id>/', views.AlertDetailAPIView.as_view(), name='alert_detail'),
# ]