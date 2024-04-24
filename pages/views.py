import os.path

from django.shortcuts import render
from collections import defaultdict
from django.http import JsonResponse
from pages.models import DirectoryRoot, Directory, EmbeddingDirectory

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader



# Create your views here.

def home_page(request, *args, **kwargs):

    # If POST request contains files
    if request.method == 'POST' and request.FILES:

        # Each uploaded directory is associated with a directory root object
        # which carries information about directory 'name', 'structure', and 'user'
        root = DirectoryRoot.objects.create(
            name=make_name(request.FILES['file_0'].name.split('___')[0]),
            user=request.user
        )

        # Reconstruct directory structure and document objects from files in request.FILE
        directory_structure = defaultdict(list)

        for i in range(len(request.FILES)):
            file = request.FILES['file_{}'.format(i)]
            path = file.name.split('___')           # An example of filename that is passed in: 'directory___subdirectory___filename'
            i = 1
            for curr, next in zip(path[:-1], path[i:]):
                if next not in directory_structure[curr]:
                    directory_structure[curr].append(next)
                i += 1
            Directory.objects.create(directory=file, root=root)     # Each document object is linked to the base directory object

        root.structure = directory_structure
        root.save()

        # Create a directory embeddings object that links to the base directory object
        embeddings_directory = EmbeddingDirectory.objects.create(name=root.name, directory=root)

        return JsonResponse({'message': 'success!'})
    return render(request, "home.html")


def fetch_directory_tree(request):
    directory_id = request.GET.get('directory_id')
    return JsonResponse({'directory': DirectoryRoot.objects.filter(id=directory_id).get().structure})


def fetch_document(request):
    """Get document text content
    """
    path = request.GET.get('source')
    path = "\\".join(path.split('___'))
    extension = path.split('.')[-1]
    if os.path.exists(path):
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
    """Create a unique directory name from the given name.
    Avoid duplication by adding a suffix if name is taken
    """
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
