from django.shortcuts import render
from pages.forms import DirectoryForm
from collections import defaultdict
from django.http import JsonResponse
from pages.models import DirectoryRoot, Directory
# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST' and request.FILES:
        root = DirectoryRoot.objects.create(name=request.FILES['file_0'].name.split('___')[0])
        directory_structure = defaultdict(list)
        for i in range(len(request.FILES)):
            file = request.FILES['file_{}'.format(i)]
            path = file.name.split('___')
            i = 1
            for curr, next in zip(path[:-1], path[i:]):
                if next not in directory_structure[curr]:
                    directory_structure[curr].append(next)
                i += 1
            Directory.objects.create(directory=file, root=root)
        return JsonResponse({'directory': directory_structure})
    return render(request, "pages/home.html")