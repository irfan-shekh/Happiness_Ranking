from django.urls import path
from .views import first, predict_ajax

urlpatterns = [
    path('', first, name='homepage'),
    path('predict/', predict_ajax, name='predict_ajax'),
]
