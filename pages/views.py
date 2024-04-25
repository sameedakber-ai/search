import os.path

from django.shortcuts import render
from collections import defaultdict
from django.http import JsonResponse
from pages.models import Directory, File, Embedding

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


def home_page(request, *args, **kwargs):
    return render(request, "home.html")


def upload_files(request):

    if request.method == 'POST':

        if request.FILES:

            directory_key = "".join(request.POST['key'].split(','))

            directory = Directory.objects.create(name=make_name(request.FILES['file_0'].name.split('___')[0]), key=directory_key)

            directory_structure = defaultdict(list)

            for i in range(len(request.FILES)):

                file = request.FILES['file_{}'.format(i)]

                path = file.name.split('___')

                i = 1
                for curr, next in zip(path[:-1], path[i:]):
                    if next not in directory_structure[curr]:
                        directory_structure[curr].append(next)
                    i += 1

                File.objects.create(file=file,directory=directory)

            directory.structure = directory_structure

            directory.save()

            embeddings = Embedding.objects.create(name=directory.name, directory=directory)

            return JsonResponse({'message': 'success!'})

        return ({'message': 'Files could not be uploaded. Please try again'})


def fetch_directory_tree(request):
    directory_id = request.GET.get('directory_id')
    return JsonResponse({'directory': Directory.objects.filter(id=directory_id).get().structure})


def fetch_document(request):
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
    return JsonResponse(
        {'document': '<p>Document does not exist or is corrupted.<br> Please do a fresh directory upload</p>'})


def make_name(name):
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
