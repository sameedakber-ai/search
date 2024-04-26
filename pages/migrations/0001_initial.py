# Generated by Django 4.2.11 on 2024-04-26 00:55

import datetime
from django.db import migrations, models
import django.db.models.deletion
import pages.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Directory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128)),
                ('key', models.CharField(max_length=128, unique=True)),
                ('structure', models.JSONField(default={}, null=True)),
                ('date', models.DateTimeField(blank=True, default=datetime.datetime.now)),
                ('chat_history', models.JSONField(default={'chat_history': []}, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to=pages.models.get_upload_path)),
                ('directory', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pages.directory')),
            ],
        ),
        migrations.CreateModel(
            name='Embedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128)),
                ('processed', models.BooleanField(default=False)),
                ('directory', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='pages.directory')),
            ],
        ),
    ]
