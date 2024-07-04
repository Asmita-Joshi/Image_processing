# image_processing_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path('download/<int:image_id>/', views.download_image, name='download_image'),
    path('result/', views.result, name='result'),
    path('', views.firstpage, name='firstpage'),
    path('login/', views.afterlogin, name='afterlogin'),
    
    
]

