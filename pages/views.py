from django.shortcuts import render
from pages.forms import DirectoryForm
# Create your views here.

def home_page(request, *args, **kwargs):
    if request.method == 'POST':
        form = DirectoryForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    form = DirectoryForm()
    context = {'form': form}
    return render(request, "pages/home.html", context=context)
