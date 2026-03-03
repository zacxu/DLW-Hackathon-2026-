from django.urls import path
from frontend import views

urlpatterns = [
    path("", views.home, name="home"),
    path("fire/", views.fire, name="fire"),
    path("smoke/", views.smoke, name="smoke"),
    path("api/infer/", views.run_inference, name="run_inference"),
    path("api/emergency/", views.contact_emergency, name="contact_emergency"),
    path("api/emergency/standalone/", views.contact_emergency_standalone, name="contact_emergency_standalone"),
    path("api/contact/", views.submit_contact_message, name="submit_contact_message"),
    path("api/route-plan/", views.plan_policy_route, name="plan_policy_route"),
]
