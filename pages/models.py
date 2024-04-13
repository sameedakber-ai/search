from django.db import models

# def content_file_name(instance, filename):
#     name, ext = filename.split('.')
#     file_path = '{account_id}/photos/user_{user_id}.{ext}'.format(
#          account_id=instance.account_id, user_id=instance.id, ext=ext)
#     return file_path

def get_upload_path(instance, filename):
    path = filename.replace('___', '/')
    file_path = 'directories/{}'.format(path)
    return file_path

class DirectoryRoot(models.Model):
    name = models.CharField(max_length=128)

# Create your models here.
class Directory(models.Model):
    directory = models.FileField(upload_to=get_upload_path)
    root = models.ForeignKey(DirectoryRoot, on_delete=models.CASCADE)
