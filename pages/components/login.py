from django_unicorn.components import UnicornView,LocationUpdate
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect



class LoginView(UnicornView):
    username = ''
    password = ''

    def login(self):
        user = authenticate(self.request, username=self.username, password=self.password)
        if user is not None:
            login(self.request, user)
            return redirect('/')

    def logout(self):
        logout(self.request)
        return redirect('/')