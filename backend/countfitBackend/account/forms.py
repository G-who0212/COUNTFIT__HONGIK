from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField # 관리자 페이지에서 사용자의 비밀번호를 표시하는 필드 / 읽기 전용 (readonly)
from django.utils.translation import gettext_lazy as _ # 문자열이 번역되는 시점을 나중으로 미룸. / Django의 번역 시스템이 템플릿 렌더링 시점에 문자열을 번역할 수 있도록 함.

from .models import User
from .managers import UserManager


class UserCreationForm(forms.ModelForm):
    # 사용자 생성 폼
    email = forms.EmailField(
        label=_('Email'),
        required=True,
        widget=forms.EmailInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Email address'),
                'required': 'True',
            }
        )
    )
    nickname = forms.CharField(
        label=_('nickname'),
        required=True,
        widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Nickname'),
                'required': 'True',
            }
        )
    )

    gender = forms.CharField(
        label=_('gender'),
        required=False,
        widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Gender'),
                'required': 'False',
            }
        )
    )

    age = forms.IntegerField(
        label=_('Age'),
        required=False,
        widget=forms.NumberInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Age'),
                'required': 'False',
            }
        )
    )

    password1 = forms.CharField(
        label=_('Password'),
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Password'),
                'required': 'True',
            }
        )
    )
    password2 = forms.CharField(
        label=_('Password confirmation'),
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': _('Password confirmation'),
                'required': 'True',
            }
        )
    )

    class Meta:
        model = User
        fields = ('email', 'nickname', 'gender', 'age')

    def clean_password2(self):
        # 두 비밀번호 입력 일치 확인
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2

    def save(self, commit=True):
        # Save the provided password in hashed format
        user = super(UserCreationForm, self).save(commit=False)
        user.email = UserManager.normalize_email(self.cleaned_data['email'])
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user


class UserChangeForm(forms.ModelForm):
    # 비밀번호 변경 폼
    password = ReadOnlyPasswordHashField( 
        label=_('Password')
    )

    class Meta:
        model = User
        fields = ('email', 'password', 'nickname', 'gender', 'age', 'is_active', 'is_admin')

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        # This is done here, rather than on the field, because the
        # field does not have access to the initial value
        return self.initial["password"]