from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import *

urlpatterns = [
    path('signup/', UserRegisterAPIView.as_view()),
    path('signin/', UserLoginAPIView.as_view()),
    path('refresh/', TokenRefreshView.as_view()), # FE에서 만료된 access Token을 보냈을 때
    path('userinfo/', UserInfoListAPIView.as_view()) # account/userinfo/?nickname=hoo
]