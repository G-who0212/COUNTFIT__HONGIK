from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Group
from django.utils.translation import gettext_lazy as _

from .forms import UserCreationForm, UserChangeForm
from .models import User


class UserAdmin(BaseUserAdmin):
    # The forms to add and change user instances
    form = UserChangeForm
    add_form = UserCreationForm

    # The fields to be used in displaying the User model.
    # These override the definitions on the base UserAdmin
    # that reference specific fields on auth.User.
    ordering = ['nickname']
    list_display = ('email', 'nickname', 'gender', 'age', 'is_active', 'is_admin') # 관리자 페이지에서 표시할 열(칼럼)들
    list_display_links = ('nickname',) # 목록 페이지에서 클릭 가능한 링크로 표시할 필드 -> 누르면 모델 인스턴스의 세부 페이지로 이동
    list_filter = ('is_superuser', 'is_active',) # 목록 페이지의 사이드바에 필터를 추가하여 데이터를 필터링할 수 있도록 함.
    fieldsets = ( # 관리자 페이지에서 사용자 수정 시, 폼에 표시할 필드들의 순서와 구성
        (None, {'fields': ('email', 'password')}),
        (_('Personal info'), {'fields': ('nickname', 'gender', 'age')}),
        (_('Permissions'), {'fields': ('is_active', 'is_admin',)}),
    )
    # add_fieldsets is not a standard ModelAdmin attribute. UserAdmin
    # overrides get_fieldsets to use this attribute when creating a user.
    add_fieldsets = ( # 관리자 페이지에서 사용자 수정 시, 폼에 표시할 필드들의 순서와 구성
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'nickname', 'password1', 'password2')}
         ),
    )
    search_fields = ('email','nickname') # 모델의 데이터 검색을 email, nickname으로 함.
    filter_horizontal = () # 다대다 관계를 가진 필드를 보다 직관적으로 나타낼때 사용


# Now register the new UserAdmin...
admin.site.register(User, UserAdmin)