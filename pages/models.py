from django.db import models
from datetime import datetime
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session

class Directory(models.Model):
    """Database table - directories"""
    name = models.CharField(max_length=128)
    key = models.CharField(max_length=128, unique=True, null=False)
    structure = models.JSONField(null=True, default={})
    date = models.DateTimeField(default=datetime.now, blank=True)
    chat_history = models.JSONField(null=True, default={'chat_history': []})

def get_upload_path(instance, filename):
    """Set storage path for files"""
    root_name = Directory.objects.order_by('-date')[0].name
    path_excluding_root = "/".join(filename.split('___')[1:])
    path = root_name + "/" + path_excluding_root
    file_path = 'directories/{0}'.format(path)
    return file_path

class File(models.Model):
    """Database table - files"""
    file = models.FileField(upload_to=get_upload_path)
    directory = models.ForeignKey(Directory, on_delete=models.CASCADE)

class Embedding(models.Model):
    """Database table - embeddings"""
    name = models.CharField(max_length=128)
    directory = models.OneToOneField(Directory, on_delete=models.CASCADE)
    processed = models.BooleanField(default=False)

