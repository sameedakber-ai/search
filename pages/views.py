import os.path

from django.shortcuts import render
from collections import defaultdict
from django.http import JsonResponse
from pages.models import Directory, File, Embedding

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader



# Create your views here.

def home_page(request, *args, **kwargs):
    """Parse incoming POST file data to create a <directory> object
    For each <directory> object, create an associated <embedding> object
    For each <directory> object, create multiple <file> objects"""

    # If POST request contains files
    if request.method == 'POST' and request.FILES:

        directory = Directory.objects.create(name=make_name(request.FILES['file_0'].name.split('___')[0]))

        # Reconstruct directory structure as a dictionary
        directory_structure = defaultdict(list)

        # Get files sent in the POST request, and create <file> objects that point to these files
        for i in range(len(request.FILES)):
            file = request.FILES['file_{}'.format(i)]
            path = file.name.split('___')           # An example of filename that is passed in: 'directory___subdirectory___filename'
            i = 1
            for curr, next in zip(path[:-1], path[i:]):
                if next not in directory_structure[curr]:
                    directory_structure[curr].append(next)
                i += 1
            File.objects.create(file=file, directory=directory)

        directory.structure = directory_structure
        directory.save()

        # Create an <embedding> object
        embeddings = Embedding.objects.create(name=directory.name, directory=directory)

        # Abstract json response to signify valid response
        return JsonResponse({'message': 'success!'})

    return render(request, "home.html")


def fetch_directory_tree(request):
    directory_id = request.GET.get('directory_id')
    return JsonResponse({'directory': Directory.objects.filter(id=directory_id).get().structure})


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
    used_names = [obj.name for obj in Directory.objects.all()]
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
