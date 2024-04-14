from django.shortcuts import render
from pages.forms import DirectoryForm
from collections import defaultdict
from django.http import JsonResponse
from pages.models import DirectoryRoot, Directory, EmbeddingDirectory
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
        root.structure = directory_structure
        root.save()
        embeddings_directory = EmbeddingDirectory.objects.create(name=root.name, directory=root)
    return render(request, "pages/home.html")

def fetch_directory_tree(request):
    directory_id = request.GET.get('directory_id')
    print(directory_id)
    return JsonResponse({'directory': DirectoryRoot.objects.filter(id=directory_id).get().structure})