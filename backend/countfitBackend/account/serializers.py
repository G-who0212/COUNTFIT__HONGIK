from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import User, Record

User = get_user_model()


from rest_framework import serializers
from .models import User

class UserRegisterSerializer(serializers.ModelSerializer):
    password2 = serializers.CharField(write_only=True)  # password2 필드 추가

    class Meta:
        model = User
        fields = ['email', 'password', 'password2', 'nickname', 'gender', 'age']  # 필요한 필드만 나열
        extra_kwargs = {
            'password': {'write_only': True}
        }

    # password와 password2가 일치하는지 검증
    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError("Passwords do not match")
        return data

    # create 메서드 수정: password2는 저장하지 않음
    def create(self, validated_data):
        validated_data.pop('password2')  # password2 필드를 제거
        user = User.objects.create_user(**validated_data)  # create_user 메서드 사용
        return user



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