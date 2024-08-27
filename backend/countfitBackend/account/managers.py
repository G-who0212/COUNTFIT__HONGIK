from django.contrib.auth.models import BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, email, nickname, gender=None, age=None, password=None, **extra_fields):
        if email is None:
            raise TypeError("Users must have a email.")
        if password is None:
            raise TypeError("Users must have a password.")
        if nickname is None:
            raise TypeError("Users must have a nickname.")

        user = self.model(
            email = self.normalize_email(email),
            nickname = nickname,
            gender = gender,
            age = age,
            **extra_fields
        )

        user.set_password(password)
        user.save()

        return user

    def create_superuser(self, email, nickname, gender=None, age=None, password=None, **extra_fields):
        if password is None:
            raise TypeError("Superuser must have a password.")
        
        user = self.create_user(email, nickname, gender, age, password, **extra_fields)
        user.is_superuser = True
        user.is_admin = True
        user.save()

        return user