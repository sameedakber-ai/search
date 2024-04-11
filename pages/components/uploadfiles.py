from django_unicorn.components import UnicornView


class UploadfilesView(UnicornView):
    files = []
    directory = ''

    def upload(self):
        print(self.directory)
