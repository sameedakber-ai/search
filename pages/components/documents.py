from django_unicorn.components import UnicornView
from pages.models import DirectoryRoot

import os
import re
import shutil
from dotenv import load_dotenv

import chromadb

from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

CHROMA_PATH = "chroma"
DOCUMENT_PATH = 'media/directories'
HEADERS_TO_SPLIT_ON = [
    ('title:', "Title"),
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
load_dotenv()


class DocumentsView(UnicornView):
    dir_path = ''
    directories = DirectoryRoot.objects.all()

    def mount(self):
        self.directories = DirectoryRoot.objects.all()

    def load_documents(self, dir_path):
        loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
        documents = loader.load()
        return documents

    def create_vector_db(self, documents, chroma_path):
        if not os.path.exists(chroma_path):
            db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
            db.persist()

    def create_db(self):
        documents = self.load_documents('media/directories/{}'.format(DirectoryRoot.objects.filter(id=1).get().name))
        vector_db = self.create_vector_db(documents, 'media/chroma')
        self.directories = DirectoryRoot.objects.all()
