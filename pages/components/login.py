from django_unicorn.components import UnicornView,LocationUpdate
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.core.exceptions import ValidationError
from django.contrib import messages



class LoginView(UnicornView):
    username = ''
    password = ''

    def login(self):
        user = authenticate(self.request, username=self.username, password=self.password)
        if user is not None:
            login(self.request, user)
            return redirect('/')
        else:
            messages.error(self.request, "Credentials do not match")

    def logout(self):
        logout(self.request)
        return redirect('/')