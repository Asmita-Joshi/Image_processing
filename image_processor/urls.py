"""
URL configuration for image_processor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import path, include

from image_processing_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('image_processing_app.urls')),
    path('accounts/', include('django.contrib.auth.urls')),  # Include Django authentication URLs
    path('signup/', views.signup, name='signup'),  # Define the URL pattern for the 'signup' view
    path('', views.index, name='index'),  # Define the URL pattern for the 'index' view
    path('result/', views.result, name='result'),  # Define the URL pattern for the 'result' view
    path('home/', views.home, name='home'), 
    path('dashboard/', views.dashboard, name='dashboard'), 
    
    
]


