from django.shortcuts import render
from pages.forms import DirectoryForm, Directory
from collections import defaultdict
from django.http import JsonResponse
# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST':
        directory_structure = defaultdict(list)
        for i in range(len(request.FILES)):
            file = request.FILES['file_{}'.format(i)]
            path = file.name.split('___')
            i = 1
            for curr, next in zip(path[:-1], path[i:]):
                if next not in directory_structure[curr]:
                    directory_structure[curr].append(next)
                i += 1
            Directory.objects.create(directory=file)
        return JsonResponse({'directory': directory_structure})
    return render(request, "pages/home.html")