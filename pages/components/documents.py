from pages.models import DirectoryRoot

from django_unicorn.components import UnicornView
from django.contrib.humanize.templatetags.humanize import naturalday
from django.contrib.auth import logout
from django.shortcuts import redirect

import os
import re
import shutil
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

loaders = {
    '.pdf': PyPDFLoader,
    '.md': TextLoader,
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
        """Create separate directory loaders
        for different file types"""
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
        sorted_directories['today'].extend(
            [directory for directory in directories if naturalday(directory.date) == 'today'])
        sorted_directories['yesterday'].extend(
            [directory for directory in directories if naturalday(directory.date) == 'yesterday'])
        sorted_directories['previous'].extend([directory for directory in directories if not (
                    directory in sorted_directories['today'] or directory in sorted_directories['yesterday'])])
        self.directories = sorted_directories

    def mount(self):
        self.initialize_directory_data()

    def load_documents(self, dir_path):
        """Load and combine documents from
        different (file type) directory loaders"""
        pdf_loader = self.create_directory_loader('.pdf', dir_path)
        md_loader = self.create_directory_loader('.md', dir_path)
        txt_loader = self.create_directory_loader('.txt', dir_path)
        docx_loader = self.create_directory_loader('.docx', dir_path)

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

    def create_vector_db(self, documents, chroma_path = '', persist=True):
        """Create and store vector database"""
        db = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
        if persist:
            db.persist()

    def get_vector_db(self, chroma_path):
        """Get stored vector database"""
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        return db

    def get_k_relevant_documents(self, db, query, k):
        """Do a vector similarity search
        to extract relevant documents"""
        return db.similarity_search_with_relevance_scores(query, k)

    def split_document(self, documents):
        """Split a document into chunks"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=400
        )
        return text_splitter.create_documents(documents)

    def create_db(self):
        """Create and persist vector
        database if it does not exist"""
        if not os.path.exists(self.selected_vector_db_path):
            documents = self.load_documents('media/{}/directories/{}'.format(self.request.user.id, self.selected_directory.name))
            vector_db = self.create_vector_db(documents, self.selected_vector_db_path)

    def get_recontextualized_question(self):
        """Add context to user question
        based on chat history"""
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
        """Feed the most relevant text
        into LLM for response"""
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

        if isinstance(self.selected_directory, dict):
            self.selected_directory = DirectoryRoot.objects.filter(name=self.selected_directory['name']).first()

        query = self.get_recontextualized_question()

        db = self.get_vector_db(self.selected_vector_db_path)

        docs = self.get_k_relevant_documents(db, query, k=3)

        relevant_docs = self.sort_docs_by_relevance_scores(docs, self.cutoff_score)

        if relevant_docs is None:
            self.call('showNoRelevantDataMessage')
        else:
            print('Total number of relevant chunks: ', len(relevant_docs))

            sources = [doc.metadata.get('source', None) for doc, _score in relevant_docs]
            scores = [round(_score, 2) for doc, _score in relevant_docs]

            chunks = []
            for doc in relevant_docs:
                chunks.extend(self.split_document([doc[0].page_content]))

            relevant_chunks = self.get_k_relevant_documents(db, query, k=5)

            relevant_chunks = self.sort_docs_by_relevance_scores(relevant_chunks, self.cutoff_score)

            if relevant_docs is None:
                self.call('showNoRelevantDataMessage')
            else:
                history = self.selected_directory.chat_history['chat_history']

                llm_response = self.get_llm_response(query, relevant_chunks, history)

                formatted_response = f"{llm_response}<div class='mt-4'>"
                for source, score in zip(sources, scores):
                    modified_source = "/".join(source.split(f"\\")[3:])
                    request_source = "___".join(source.split(f"\\"))
                    formatted_response += f'<a href="" class="font-bold block text-emerald-600 mb-2" onclick="showDocument(\'{request_source}\', event, this)">{modified_source} - {score}</a>'
                formatted_response += '</div>'

                self.selected_directory.chat_history['chat_history'].append([self.question, formatted_response])
                self.selected_directory.save()

                self.call("scrollToBottom")

        self.question = ''
        self.initialize_directory_data()

    def update_chat_selection(self, directory_id):
        """Create new vector database when user processes uploaded documents OR
        Get existing database"""
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
        """Relevance criteria for documents"""
        if len(documents) == 1:
            return documents
        scores = [round(_score, 2) for doc, _score in documents]
        scores.sort(reverse=True)
        best = []
        if scores[0] <= float(cutoff_score):
            return None

        for i in range(len(scores) - 1):
            curr = scores[0]
            next = scores[i + 1]
            best.append(documents[i])
            if ((curr - next) / curr) >= 0.18:
                break

        return best

    def increment(self):
        """Increase similarity threshold"""
        cutoff_score = float(self.cutoff_score)
        if cutoff_score < 0.9:
            cutoff_score += 0.1
        self.cutoff_score = round(cutoff_score, 1)
        self.call('increment')
        self.initialize_directory_data()

    def decrement(self):
        """Decrease similarity threshold"""
        cutoff_score = float(self.cutoff_score)
        if cutoff_score > 0.3:
            cutoff_score -= 0.1
        self.cutoff_score = round(cutoff_score, 1)
        self.call('decrement')
        self.initialize_directory_data()
