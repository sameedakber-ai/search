import json
import shutil

from django_unicorn.components import UnicornView
from pages.models import DirectoryRoot

import os
from dotenv import load_dotenv


from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()


class DocumentsView(UnicornView):
    dir_path = ''
    directories = []
    question = ''
    selected_directory = ''
    selected_vector_db_path = ''

    def mount(self):
        self.directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')

    def load_documents(self, dir_path):
        loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True)
        documents = loader.load()
        return documents

    def create_vector_db(self, documents, chroma_path):
        db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
        db.persist()

    def get_vector_db(self, chroma_path):
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        return db

    def get_k_relevant_documents(self, db, query, k=2):
        return db.similarity_search_with_relevance_scores(query, k)

    def create_db(self):
        if not os.path.exists(self.selected_vector_db_path):
            documents = self.load_documents('media/{}/directories/{}'.format(self.request.user.id, self.selected_directory.name))
            vector_db = self.create_vector_db(documents, self.selected_vector_db_path)
        # self.directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')

    def get_conversation_chain(self, db):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def get_recontextualized_question(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
        {chat_history}
        {question}
        """
        prompt_template = ChatPromptTemplate.from_template(contextualize_q_system_prompt)
        prompt = prompt_template.format(chat_history=self.selected_directory.chat_history['chat_history'], question=self.question)
        model = ChatOpenAI()
        response_text = model.predict(prompt)
        return response_text

    def get_llm_response(self, query, results, history):
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
        response = model.predict(prompt)

        return response

    def respond(self):
        db = self.get_vector_db(chroma_path=self.selected_vector_db_path)
        if isinstance(self.selected_directory, dict):
            self.selected_directory = DirectoryRoot.objects.filter(name=self.selected_directory['name']).first()
        query = self.get_recontextualized_question()
        print(query)
        relevant_docs = self.get_k_relevant_documents(db, query, k=2)
        sources = [doc.metadata.get('source', None) for doc, _score in relevant_docs]
        history = self.selected_directory.chat_history['chat_history']
        llm_response = self.get_llm_response(query, relevant_docs, history)
        formatted_response = f"{llm_response}<div class='mt-4'>"
        for source in sources:
            modified_source = "/".join(source.split(f"\\")[3:])
            request_source = "___".join(source.split(f"\\"))
            formatted_response += f'<a href="" class="font-bold block text-emerald-600 mb-2" onclick="showDocument(\'{request_source}\', event)">{modified_source}</a>'
        formatted_response += '</div>'
        self.selected_directory.chat_history['chat_history'].append([self.question, formatted_response])
        self.selected_directory.save()

    def update_chat_selection(self, directory_id):
        self.selected_directory = DirectoryRoot.objects.get(id=directory_id)
        self.selected_vector_db_path = 'media/{}/chroma/{}'.format(self.request.user.id, self.selected_directory.embeddingdirectory.name)
        self.create_db()
        self.directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')
        self.selected_directory.embeddingdirectory.processed = True
        self.selected_directory.embeddingdirectory.save()

    def refreshDirectories(self):
        self.directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')

    def delete(self, directory_id):
        dir_name = DirectoryRoot.objects.get(id=directory_id).name
        DirectoryRoot.objects.get(id=directory_id).delete()
        shutil.rmtree('media/{}/chroma/{}'.format(self.request.user.id, dir_name))
        shutil.rmtree('media/{}/directories/{}'.format(self.request.user.id, dir_name))
        self.selected_directory = ''
        self.selected_vector_db_path = ''
        self.directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')
