from django.urls import path
from django.contrib.auth import views as auth_views
from .views import home, signup_view, login_view, classify_list, classify_create, classify_detail,logout_view

urlpatterns = [
    path('', home, name='home'),
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('classify/', classify_list, name='classify_list'),
    path('classify/new/', classify_create, name='classify_create'),
    path('classify/<int:pk>/', classify_detail, name='classify_detail'),
]
