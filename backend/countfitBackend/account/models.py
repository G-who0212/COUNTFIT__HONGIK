from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.db.models.fields import BooleanField

from .managers import UserManager
# Create your models here.

class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(max_length=255, unique=True)
    nickname = models.CharField(max_length=100)
    gender = models.CharField(max_length=50, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    is_active = BooleanField(default=True)
    is_admin = BooleanField(default=False)

    USERNAME_FIELD = 'email'

    REQUIRED_FIELDS = ['nickname']

    objects = UserManager()

    def __str__(self):
        return self.nickname
    
    @property
    def is_staff(self):
        return self.is_admin