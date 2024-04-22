import json
import shutil

from django_unicorn.components import UnicornView
from pages.models import DirectoryRoot

import os
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader, DirectoryLoader, PyPDFLoader, \
    CSVLoader, Docx2txtLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from collections import defaultdict
from django.contrib.humanize.templatetags.humanize import naturalday

from django.contrib.auth import logout
from django.shortcuts import redirect

load_dotenv()

loaders = {
            '.pdf': PyPDFLoader,
            '.md': TextLoader,
            '.csv': CSVLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader
        }

class DocumentsView(UnicornView):
    dir_path = ''
    directories = {}
    question = ''
    selected_directory = ''
    selected_vector_db_path = ''
    cutoff_score = 0.6

    def create_directory_loader(self, file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
            silent_errors=True
        )

    def initialize_directory_data(self):
        directories = DirectoryRoot.objects.filter(user_id=self.request.user.id).order_by('-date')
        if not directories:
            return
        sorted_directories = defaultdict(list)
        sorted_directories['today'].extend([directory for directory in directories if naturalday(directory.date) == 'today'])
        sorted_directories['yesterday'].extend([directory for directory in directories if naturalday(directory.date) == 'yesterday'])
        sorted_directories['previous'].extend([directory for directory in directories if not (directory in sorted_directories['today'] or directory in sorted_directories['yesterday'])])
        self.directories = sorted_directories


    def mount(self):
        self.initialize_directory_data()

    def load_documents(self, dir_path):
        pdf_loader = self.create_directory_loader('.pdf', dir_path)
        md_loader = self.create_directory_loader('.md', dir_path)
        txt_loader = self.create_directory_loader('.txt', dir_path)
        docx_loader = self.create_directory_loader('.docx', dir_path)

        # loader = DirectoryLoader(dir_path, glob="**/*", loader_cls=TextLoader, silent_errors=True)

        all_documents = []

        pdf_documents = pdf_loader.load()
        md_documents = md_loader.load()
        txt_documents = txt_loader.load()
        docx_documents = docx_loader.load()

        all_documents.extend(pdf_documents)
        all_documents.extend(md_documents)
        all_documents.extend(txt_documents)
        all_documents.extend(docx_documents)

        return all_documents

    def create_vector_db(self, documents, chroma_path, persist=True):
        db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
        if persist:
            db.persist()

    def get_vector_db(self, chroma_path):
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        return db

    def get_k_relevant_documents(self, db, query, k=2):
        return db.similarity_search_with_relevance_scores(query, k)

    def split_document(self, document):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=400
        )
        return text_splitter.create_documents([document])

    def create_db(self):
        if not os.path.exists(self.selected_vector_db_path):
            documents = self.load_documents(
                'media/{}/directories/{}'.format(self.request.user.id, self.selected_directory.name))
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
        contextualize_q_system_prompt = """Given a labeled chat history between you and the user, and the latest user question \
        which might reference the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is. If the question implicitly refers to something in the chat history, \
        make sure to explicitly state that information in the reformulated question. Do not change the question wording if it does not \
        relate to the previous chat, return it as is.
        Chat History: {chat_history}
        Question: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(contextualize_q_system_prompt)
        chat_history = ""
        for user, llm in self.selected_directory.chat_history['chat_history'][-3:]:
            chat_history += 'USER: {}\nYOU: {}\n\n'.format(user, llm)
        prompt = prompt_template.format(chat_history=chat_history,
                                        question=self.question)
        model = ChatOpenAI()
        response_text = model.predict(prompt)
        return response_text

    def get_llm_response(self, query, results, history):
        PROMPT_TEMPLATE = """
            You are a tech assistant for an IT department. Answer the question based only on the following context:
            {context}
            ---
            Answer the question based on the above context, and format your answer as a neat html code with tags included;
            keep the outermost tag a <div>; use tailwind css styling on all tags: {question}
            """
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)

        model = ChatOpenAI()
        response = model.predict(prompt)
        pattern = re.compile(r'\b(?:bg|text)-\w+\s*')
        response = re.sub(pattern, '', response)

        return response

    def respond(self):

        if not self.cutoff_score:
            self.cutoff_score = 0.60

        if not str(self.cutoff_score).isdigit():
            self.cutoff_score = 0.60

        if float(self.cutoff_score) < 0.50 or float(self.cutoff_score) > 0.90:
            self.cutoff_score = 0.50

        db = self.get_vector_db(chroma_path=self.selected_vector_db_path)

        if isinstance(self.selected_directory, dict):
            self.selected_directory = DirectoryRoot.objects.filter(name=self.selected_directory['name']).first()

        query = self.get_recontextualized_question()
        print(query)

        docs = self.get_k_relevant_documents(db, query, k=5)
        relevant_docs = self.sort_docs_by_relevance_scores(docs, self.cutoff_score)

        if relevant_docs is None:
            self.call('showNoRelevantDataMessage')
        else:
            print('Total number of relevant files: ', len(relevant_docs))

            sources = [doc.metadata.get('source', None) for doc, _score in relevant_docs]
            scores = [round(_score, 2) for doc, _score in relevant_docs]

            split_documents = []
            for doc in relevant_docs:
                split_documents.extend(self.split_document(doc[0].page_content))

            db = Chroma.from_documents(documents=split_documents, embedding=OpenAIEmbeddings())
            docs = self.get_k_relevant_documents(db, query, k=4)
            relevant_docs = self.sort_docs_by_relevance_scores(docs, self.cutoff_score)

            if relevant_docs is None:
                self.call('showNoRelevantDataMessage')
            else:
                print('Total number of relevant chunks: ', len(relevant_docs))

                history = self.selected_directory.chat_history['chat_history']
                llm_response = self.get_llm_response(query, relevant_docs, history)

                formatted_response = f"{llm_response}<div class='mt-4'>"
                for source, score in zip(sources, scores):
                    modified_source = "/".join(source.split(f"\\")[3:])
                    request_source = "___".join(source.split(f"\\"))
                    formatted_response += f'<a href="" class="font-bold block text-emerald-600 mb-2" onclick="showDocument(\'{request_source}\', event, this)">{modified_source} - {score}</a>'
                formatted_response += '</div>'

                self.selected_directory.chat_history['chat_history'].append([self.question, formatted_response])
                self.selected_directory.save()
                self.call("scrollToBottom")
        self.initialize_directory_data()

    def update_chat_selection(self, directory_id):
        self.selected_directory = DirectoryRoot.objects.get(id=directory_id)
        self.selected_vector_db_path = 'media/{}/chroma/{}'.format(self.request.user.id,
                                                                   self.selected_directory.embeddingdirectory.name)
        self.create_db()
        self.initialize_directory_data()
        self.selected_directory.embeddingdirectory.processed = True
        self.selected_directory.embeddingdirectory.save()
        self.call("scrollToBottom")

    def refreshDirectories(self):
        self.initialize_directory_data()

    def delete(self, directory_id):
        dir_name = DirectoryRoot.objects.get(id=directory_id).name
        DirectoryRoot.objects.get(id=directory_id).delete()
        shutil.rmtree('media/{}/chroma/{}'.format(self.request.user.id, dir_name))
        shutil.rmtree('media/{}/directories/{}'.format(self.request.user.id, dir_name))
        self.selected_directory = ''
        self.selected_vector_db_path = ''
        self.initialize_directory_data()

    def logout_user(self):
        logout(self.request)
        return redirect('/')

    def sort_docs_by_relevance_scores(self, documents, cutoff_score):
        scores = [round(_score, 2) for doc, _score in documents]
        scores.sort(reverse=True)
        best = []
        if scores[0] <= float(cutoff_score):
            return None

        for i in range(len(scores) - 1):
            curr = scores[i]
            next = scores[i+1]
            best.append(documents[i])
            if ((curr-next)/curr) >= 0.3333:
                break

        return best


