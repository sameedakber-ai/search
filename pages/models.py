from django.db import models

# Create your models here.
class Directory(models.Model):
    directory = models.FileField(upload_to='directories')