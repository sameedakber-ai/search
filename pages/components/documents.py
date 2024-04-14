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
    directories = DirectoryRoot.objects.order_by('-date')
    question = "How do I do this?"
    conversation = []
    answer = ''
    selected_directory = ''
    selected_vector_db_path = ''

    def mount(self):
        self.directories = DirectoryRoot.objects.order_by('-date')
        if self.directories:
            self.selected_directory = DirectoryRoot.objects.filter(id=1).get()
            self.selected_vector_db_path = 'media/chroma/{}'.format(self.selected_directory.embeddingdirectory.name)

    def load_documents(self, dir_path):
        loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
        documents = loader.load()
        return documents

    def create_vector_db(self, documents, chroma_path):
        if not os.path.exists(chroma_path):
            db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
            db.persist()

    def create_db(self):
        documents = self.load_documents('media/directories/{}'.format(self.selected_directory.name))
        vector_db = self.create_vector_db(documents, self.selected_vector_db_path)
        self.directories = DirectoryRoot.objects.all()

    def get_vector_db(self, chroma_path):
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        return db

    def get_k_relevant_documents(self, db, query, k=1):
        return db.similarity_search_with_relevance_scores(query, k)

    def get_llm_response(self, query, results):
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

    def respond(self):
        db = self.get_vector_db(chroma_path=self.selected_vector_db_path)
        relevant_docs = self.get_k_relevant_documents(db, self.question, k=1)
        sources = [doc.metadata.get('source', None) for doc, _score in relevant_docs]
        llm_response = self.get_llm_response(self.question, relevant_docs)
        formatted_response = f"Response: {llm_response}\nSources: {sources}"
        self.answer = formatted_response
        self.conversation.append((self.question, self.answer))

    def update_chat_selection(self, directory_id):
        print(directory_id)
        self.selected_directory = DirectoryRoot.objects.filter(id=directory_id).get()
        self.selected_vector_db_path = 'media/chroma/{}'.format(self.selected_directory.embeddingdirectory.name)
        self.create_db()
