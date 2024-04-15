from django.db import models
from datetime import datetime
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session

# def content_file_name(instance, filename):
#     name, ext = filename.split('.')
#     file_path = '{account_id}/photos/user_{user_id}.{ext}'.format(
#          account_id=instance.account_id, user_id=instance.id, ext=ext)
#     return file_path

class DirectoryRoot(models.Model):
    name = models.CharField(max_length=128)
    structure = models.JSONField(null=True)
    date = models.DateTimeField(default=datetime.now, blank=True)
    chat_history = models.JSONField(null=True, default={'chat_history': []})
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, blank=True, null=True)

def get_upload_path(instance, filename):
    root_name = DirectoryRoot.objects.order_by('-date')[0].name
    path_excluding_root = "/".join(filename.split('___')[1:])
    path = root_name + "/" + path_excluding_root
    file_path = '{0}/directories/{1}'.format(instance.root.user.id, path)
    return file_path

# Create your models here.
class Directory(models.Model):
    directory = models.FileField(upload_to=get_upload_path)
    root = models.ForeignKey(DirectoryRoot, on_delete=models.CASCADE)

class EmbeddingDirectory(models.Model):
    name = models.CharField(max_length=128)
    directory = models.OneToOneField(DirectoryRoot, on_delete=models.CASCADE)
    processed = models.BooleanField(default=False)

