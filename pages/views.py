import os.path

from django.shortcuts import render
from pages.forms import DirectoryForm
from collections import defaultdict
from django.http import JsonResponse
from pages.models import DirectoryRoot, Directory, EmbeddingDirectory
import pypdf
import mammoth
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, DirectoryLoader, PyPDFLoader, \
    CSVLoader, Docx2txtLoader, PyMuPDFLoader

style_map = """
p[style-name='Section Title'] => h1:fresh
p[style-name='Subsection Title'] => h2:fresh
u => em
"""

# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST' and request.FILES:
        root = DirectoryRoot.objects.create(
            name=make_name(request.FILES['file_0'].name.split('___')[0]),
            user=request.user
        )
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
        return JsonResponse({'message': 'success!'})
    return render(request, "home.html")


def fetch_directory_tree(request):
    directory_id = request.GET.get('directory_id')
    return JsonResponse({'directory': DirectoryRoot.objects.filter(id=directory_id).get().structure})


def fetch_document(request):
    path = request.GET.get('source')
    path = "\\".join(path.split('___'))
    extension = path.split('.')[-1]
    if os.path.exists(path):
        loader = TextLoader(path)
        if extension == 'txt' or extension == 'md':
            loader = TextLoader(path)
        elif extension == 'pdf':
            loader = PyPDFLoader(path)
        elif extension == 'docx':
            loader = Docx2txtLoader(path)
        else:
            return JsonResponse({'document': '<p>UnSupported File Type</p>'})
        document = loader.load()
        return JsonResponse({'document': document[0].page_content})
    return JsonResponse({'document': '<p>Document does not exist or is corrupted.<br> Please do a fresh directory upload</p>'})

def make_name(name):
    used_names = [obj.name for obj in DirectoryRoot.objects.all()]
    if name not in used_names:
        return name
    else:
        suffix = 1
        name_is_used = True
        new_name = name
        while (name_is_used):
            new_name = name + '_' + str(suffix)
            suffix += 1
            if new_name not in used_names:
                name_is_used = False
        return new_name
