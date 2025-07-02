from django.urls import path
from detectors import views

urlpatterns = [
    path('', views.get_email, name='get_email'),      # Email subscription page at /
    path('analyze/', views.index, name='index'),      # News analyze page at /analyze/
]
