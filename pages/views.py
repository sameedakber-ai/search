from django.shortcuts import render
from pages.forms import DirectoryForm, Directory
# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST':
        for i in range(len(request.FILES)):
            file = request.FILES['file_{}'.format(i)]
            Directory.objects.create(directory=file)
    return render(request, "pages/home.html")