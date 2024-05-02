import json

from pages.models import Directory, File

from django_unicorn.components import UnicornView
from django.contrib.humanize.templatetags.humanize import naturalday

import os
import re
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

import math

load_dotenv()

loaders = {
    'pdf': PyPDFLoader,
    'md': TextLoader,
    'txt': TextLoader,
    'docx': Docx2txtLoader
}

embedding_models = {

    'bge': HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    ),

    'openai': OpenAIEmbeddings()
}


class DocumentsView(UnicornView):
    dir_path = ''

    directories = {}

    question = ''

    selected_directory = ''

    cutoff_score = 0.6

    embedding_model = 'bge'


    def initialize_directory_data(self):

        directories = Directory.objects.order_by('-date')

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

    def load_file(self, file_path):

        extension = os.path.splitext(file_path)[1].lstrip('.').lower()

        document = ''

        documents = []

        try:

            loader = loaders[extension]

        except KeyError:

            print("Unknown file extension ... skipping file")

        else:

            try:

                if loader == TextLoader:

                    document = loader(file_path, encoding="UTF-8").load()

                else:

                    document = loader(file_path).load()

            except FileNotFoundError:

                print("File not found ... skipping file")

            except RuntimeError:

                print("Can not read file ... skipping file")

            else:

                # If file is markdown, split by markdown headers and combine these into sets of markdown sections

                headers_to_split_on = [
                    ("##", "Header"),
                ]

                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,
                                                               strip_headers=False)

                md_header_splits = markdown_splitter.split_text(document[0].page_content)

                md_header_splits_batch_size = 20

                documents = []

                for i in range(0, len(md_header_splits), md_header_splits_batch_size):
                    md_header_splits_batch = md_header_splits[i:i + md_header_splits_batch_size]
                    md_header_splits_batch_text_content = '\n\n'.join([md_header_section.page_content for md_header_section in md_header_splits_batch])
                    document = Document(page_content=md_header_splits_batch_text_content, metadata={'source': file_path})
                    documents.append(document)

                print(len(documents))


                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=100000,
                    chunk_overlap=5000,
                    length_function=len,
                    is_separator_regex=False
                )

        return documents


    def create_vector_db(self, documents, chroma_path='', persist=True):

        embedding = embedding_models[self.embedding_model]

        db = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=chroma_path)

        if persist:

            db.persist()

        return db

    def get_vector_db(self, chroma_path):

        embedding_function = embedding_models[self.embedding_model]

        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

        return db

    def get_k_relevant_documents(self, db, query, k):

        return db.similarity_search_with_relevance_scores(query, k)

    def split_document(self, documents):

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000,
            chunk_overlap=200
        )

        headers_to_split_on = [
            ("##", "Header"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        splits = markdown_splitter.split_text(documents[0])

        return splits

    def create_db(self, directory):

        vector_db_path = "media/embeddings/{}".format(directory.name)

        if not directory.processed:

            channel_layer = get_channel_layer()

            files = File.objects.filter(directory=directory, processed=False).all()

            async_to_sync(channel_layer.group_send)(

                "upload",
                {
                    "type": "send_process_message",
                    "id": directory.id,
                    "uploaded": "0 / {0}".format(len(files)),
                    "progress": '<p>|<span class="text-gray-400">' + "=" * 10 + '|</span></p>',
                }

            )

            file_batch_size = int(math.ceil(len(files) / 5))

            file_batches = [files[i:i + file_batch_size] for i in range(0, len(files), file_batch_size)]

            unsuccessful_file_load_count = 0

            for i, file_batch in enumerate(file_batches):

                loaded_split_files = [self.load_file('media/{}'.format(file.file)) for file in file_batch]

                loaded_split_files = [loaded_split_file for loaded_split_file in loaded_split_files if loaded_split_file]

                loaded_files = []

                for loaded_split_file in loaded_split_files:
                    loaded_files.extend(loaded_split_file)

                unsuccessful_file_load_count += len(file_batch) - len(loaded_split_files)

                if loaded_files:

                    loaded_file_batch_size = int(math.ceil(len(loaded_files) / 5))

                    loaded_file_batches = [loaded_files[i:i + loaded_file_batch_size] for i in range(0, len(loaded_files), loaded_file_batch_size)]

                    for j, loaded_file_batch in enumerate(loaded_file_batches):

                        vector_db = self.create_vector_db(loaded_file_batch, vector_db_path)

                        current_progress = int(math.ceil((j + 1) * 10 * loaded_file_batch_size / len(loaded_files)))

                        async_to_sync(channel_layer.group_send)(

                            "upload",
                            {
                                "type": "send_sub_process_message",
                                "id": directory.id,
                                "uploaded": "{0} / {1}".format((j + 1) * loaded_file_batch_size, len(loaded_files)),
                                "progress": '<p>|' + "=" * current_progress + '<span class="text-gray-400">' + "=" * (
                                            10 - current_progress) + '|</span></p>',
                            }

                        )

                for file in file_batch:
                    file.processed = True
                    file.save()

                current_progress = int(math.ceil((i + 1) * 10 * file_batch_size / len(files)))

                async_to_sync(channel_layer.group_send)(

                    "upload",
                    {
                        "type": "send_process_message",
                        "id": directory.id,
                        "uploaded": "{0} / {1}".format((i + 1) * file_batch_size, len(files)),
                        "progress": '<p>|' + "=" * current_progress + '<span class="text-gray-400">' + "=" * (10 - current_progress) + '|</span></p>',
                    }

                )

            print("Unsuccessful file load count: ", unsuccessful_file_load_count)

            if File.objects.filter(directory=directory, processed=False).count() == 0:
                directory.processed = True
                directory.save()

        self.call('enable')



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

        prompt = prompt_template.format(chat_history=chat_history, question=self.question)

        model = ChatOpenAI()

        response_text = model.predict(prompt)

        return response_text

    def get_llm_response(self, query, results):

        PROMPT_TEMPLATE = """
            You are a tech assistant for an IT department. Answer the question based only on the following context:
            {context}
            ---
            Answer the question based on the above context, and format your answer as a neat html code with tags included;
            keep the outermost tag a <div>; use tailwind css styling on all tags: {question}
            """

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        print("Total number of tokens in context: ",
              len(context_text) / 3)  # Each token is approximately 3 characters (from openai documentation)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        prompt = prompt_template.format(context=context_text, question=query)

        model = ChatOpenAI()

        response = model.predict(prompt)

        pattern = re.compile(r'\b(?:bg|text)-\w+\s*')

        response = re.sub(pattern, '', response)

        return response

    def respond(self):

        if isinstance(self.selected_directory, dict):
            self.selected_directory = Directory.objects.filter(name=self.selected_directory['name']).first()

        directory = self.selected_directory

        vector_db_path = 'media/embeddings/{}'.format(directory.name)

        query = self.question

        vector_db_from_all_files = self.get_vector_db(vector_db_path)

        files = self.get_k_relevant_documents(vector_db_from_all_files, query, k=10)

        relevant_files = self.sort_docs_by_relevance_scores(files, self.cutoff_score)

        if relevant_files is None:
            self.call('showNoRelevantDataMessage')
        else:
            print('Total number of relevant files: ', len(relevant_files))

            sources = [doc.metadata.get('source', None) for doc, _score in relevant_files]
            scores = [round(_score, 2) for doc, _score in relevant_files]

            text_chunks = []
            for file in relevant_files:
                text_chunks.extend(self.split_document([file[0].page_content]))

            vector_db_from_relevant_chunks_only = self.create_vector_db(documents=text_chunks, chroma_path='', persist=False)

            relevant_text_chunks = self.get_k_relevant_documents(vector_db_from_relevant_chunks_only, query, k=5)

            relevant_text_chunks_based_on_criteria = self.sort_docs_by_relevance_scores(relevant_text_chunks,
                                                                                        self.cutoff_score)

            if relevant_text_chunks_based_on_criteria is None:
                self.call('showNoRelevantDataMessage')
            else:

                print("Total number of relevant chunks: ", len(relevant_text_chunks_based_on_criteria))

                llm_response = self.get_llm_response(query, relevant_text_chunks_based_on_criteria)

                sources_dict = defaultdict(list)
                for source, score in zip(sources, scores):
                    sources_dict[source].append(score)

                formatted_response = f"{llm_response}<div class='mt-4'>"
                for source in sources_dict:
                    modified_source = source.split(f"/")[-1]
                    formatted_response += f'<p class="font-bold block text-emerald-600 mb-2">{modified_source} - {max(sources_dict[source])}</p>'
                formatted_response += '</div>'

                directory.chat_history['chat_history'].append([self.question, formatted_response])

                directory.save()

                self.call("scrollToBottom")

        self.question = ''

        self.initialize_directory_data()

        self.call('enable')

    def update_chat_selection(self, directory_id):

        directory = Directory.objects.get(id=directory_id)

        if directory.processed:

            self.selected_directory = directory

            self.call("scrollToBottom")

        else:

            self.create_db(directory)

        self.initialize_directory_data()


    def sort_docs_by_relevance_scores(self, documents, cutoff_score):

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
            if ((curr - next) / curr) >= 0.05:
                break

        return best

    def increment(self):

        cutoff_score = float(self.cutoff_score)

        if cutoff_score < 0.9:
            cutoff_score += 0.1

        self.cutoff_score = round(cutoff_score, 1)

        self.call('increment')

        self.initialize_directory_data()

    def decrement(self):

        cutoff_score = float(self.cutoff_score)

        if cutoff_score > 0.3:
            cutoff_score -= 0.1

        self.cutoff_score = round(cutoff_score, 1)

        self.call('decrement')

        self.initialize_directory_data()

    def refreshDirectories(self):
        self.initialize_directory_data()