from django.urls import path
from . import views 

urlpatterns = [
    path('',views.home,name = 'home-page'),
    path('media/',views.convert,name = 'convert-photo'),
]