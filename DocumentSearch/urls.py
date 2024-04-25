"""
URL configuration for DocumentSearch project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from pages.views import home_page, fetch_directory_tree, fetch_document, upload_files

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_page, name="home"),
    path("unicorn/", include("django_unicorn.urls")),
    path('fetch_directory_tree/', fetch_directory_tree, name="fetch_directory_tree"),
    path('fetch_document/', fetch_document, name="fetch_document"),
    path('upload_files/', upload_files)
]
