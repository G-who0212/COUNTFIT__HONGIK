from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import User, Record

User = get_user_model()


class UserRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserLoginSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('nickname', 'gender', 'age')

class RecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Record
        fields = '__all__'

class UserInfoListSerializer(serializers.ModelSerializer):
    record = RecordSerializer(many=True, read_only=True)

    class Meta:
        model = User
        fields = ['nickname', 'gender', 'age']