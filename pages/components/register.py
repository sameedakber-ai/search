from django_unicorn.components import UnicornView, LocationUpdate
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.contrib import messages


class RegisterView(UnicornView):
    username = ''
    email = ''
    password = ''
    authenticated = True

    def register(self):

        self.authenticated = True

        user = User.objects.filter(email=self.email)

        if len(self.username) <= 4:
            messages.error(self.request, "Username: minimum 5 characters")
            self.authenticated = False

        if len(self.password) <= 6:
            messages.error(self.request, "Password: minimum 6 characters")
            self.authenticated = False

        if User.objects.filter(username=self.username):
            messages.error(self.request, "Username: username is taken")
            self.authenticated = False

        elif user:
            messages.error(self.request, "Email: user already exists")
            self.authenticated = False

        if self.authenticated:
            user = User.objects.create_user(username=self.username, email=self.email, password=self.password)
            login(self.request, user)
            return redirect('/')
