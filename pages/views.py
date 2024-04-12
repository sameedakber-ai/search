from django.shortcuts import render
from pages.forms import DirectoryForm, Directory
# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST' and request.FILES['file']:
        Directory.objects.create(directory=request.FILES['file'])
        print(request.FILES['file'])
    return render(request, "pages/home.html")