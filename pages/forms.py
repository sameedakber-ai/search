from django import forms
from pages.models import Directory

class DirectoryForm(forms.ModelForm):
    class Meta:
        model = Directory
        fields = ["directory",]