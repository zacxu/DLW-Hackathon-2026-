from django.urls import path
from frontend import views

urlpatterns = [
    path("", views.base, name="base"),
    path("fire", views.bitcoin, name="fire"),
    path("smoke", views.ethereum, name="smoke"),
]