from django_unicorn.components import UnicornView,LocationUpdate
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect


class RegisterView(UnicornView):
    username = ''
    email = ''
    password = ''

    def register(self):
        user = User.objects.filter(username=self.username)
        if not user:
            user = User.objects.create_user(username=self.username, email=self.email, password=self.password)
            login(self.request, user)
            return redirect('/')
