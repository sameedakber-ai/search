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


class ProcessView(UnicornView):
    dir_path=''

    def load_documents(self, dir_path):
        loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
        documents = loader.load()
        return documents

    def create_vector_db(self, documents, chroma_path):
        if not os.path.exists(chroma_path):
            db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
            db.persist()

    def get_vector_db(chroma_path):
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        return db

    def get_k_relevant_documents(db, query, k=1):
        return db.similarity_search_with_relevance_scores(query, k)

    def get_llm_response(query, results):
        PROMPT_TEMPLATE = """
        You are a tech assistant for an IT department. Answer the question based only on the following context:
        {context}
        ---
        Answer the question based on the above context: {question}
        """
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        model = ChatOpenAI()
        response_text = model.predict(prompt)
        return response_text

    def create_db(self):
        documents = self.load_documents('media/directories/{}'.format(DirectoryRoot.objects.filter(id=1).get().name))
        vector_db = self.create_vector_db(documents, 'media/chroma')