import json

from pages.models import Directory, File

from django_unicorn.components import UnicornView
from django.contrib.humanize.templatetags.humanize import naturalday

import os
import re
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
import tiktoken


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

                if extension == 'md':

                    segments = [segment.strip() for segment in re.split(r'(?=---\ntitle:)', document[0].page_content, flags=re.MULTILINE) if segment]

                    documents = []

                    for segment in segments:

                        title_pattern = r'(?<=title:\s)(.*?)(?=\n)'

                        title_matches = re.search(title_pattern, segment)

                        title = title_matches[0].strip() if title_matches else ""

                        title_summary = self.get_first_paragraph_from_text(segment, type='title')

                        split_segments = re.split(r'(?=^##\s)', segment, flags=re.MULTILINE)

                        split_segments = [segment.strip() for segment in split_segments]

                        documents.append(Document(page_content="Document Summary:\n" + title_summary + "\n\n" + "Full Document:\n" + split_segments[0], metadata={'source': file_path, 'headers': title}))

                        if len(split_segments) > 1:

                            for split_segment in split_segments[1:]:

                                split_sub_segments = re.split(r'(?=^###\s)', split_segment, flags=re.MULTILINE)

                                split_sub_segments = [segment.strip() for segment in split_sub_segments]

                                header_summary = ""

                                for split_sub_segment in split_sub_segments:

                                    header_summary += self.get_first_paragraph_from_text(split_sub_segment, type='subheader') + "\n"

                                pattern = r'(?<=##\s).*'

                                header_matches = re.finditer(pattern, split_segment)

                                headers = "\n".join([header.group() for header in header_matches if header_matches])

                                page_content = "Document Summary:\n" + header_summary[1:]

                                page_content += "\n\n" + "Full Document:\n" + split_segment

                                document = Document(page_content=page_content,
                                                    metadata={'source': file_path,
                                                              'headers': headers})

                                documents.append(document)

                else:

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=20000,
                        chunk_overlap=1000,
                        length_function=len,
                        is_separator_regex=False
                    )

                    documents = [Document(page_content=split, metadata={'source': file_path, 'headers': ''}) for split in text_splitter.split_text(document[0].page_content)]

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

    def split_document(self, document, file_path):

        text = document[0].page_content.split('Full Document:\n')[1]

        extension = os.path.splitext(file_path)[1].lstrip('.').lower()

        documents = []

        title_pattern = r'(?<=title:\s)(.*?)(?=\n)'

        title_matches = re.search(title_pattern, text)

        title = title_matches[0].strip() if title_matches else ""

        if extension == 'md':

            split_segments = re.split(r'(?=^###\s)', text, flags=re.MULTILINE)

            split_segments = [segment.strip() for segment in split_segments]

            for split_segment in split_segments:

                pattern = r'(?<=###\s).*'

                header_matches = re.finditer(pattern, split_segment)

                headers = "\n".join([header.group() for header in header_matches if header_matches])

                if title:
                    headers = title

                split = Document(page_content=split_segment,
                                    metadata={'source': file_path,
                                              'headers': headers})

                documents.append(split)

        else:

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False
            )

            documents = [Document(page_content=split, metadata={'source': file_path, 'headers': ''}) for split in text_splitter.split_text(document)]

        return documents

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

            file_batch_size = int(math.ceil(len(files) / 10))

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

                    async_to_sync(channel_layer.group_send)(

                        "upload",
                        {
                            "type": "send_sub_process_message",
                            "id": directory.id,
                            "uploaded": "0 / {}".format(len(loaded_files)),
                            "progress": '<p>|<span class="text-gray-400">' + "=" * 10 + '|</span></p>',
                        }

                    )

                    loaded_file_batch_size = int(math.ceil(len(loaded_files) / 10))

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

                async_to_sync(channel_layer.group_send)(

                    "upload",
                    {
                        "type": "checkmark_files",
                        "ids": ["check-{}".format(str(file.file)) for file in file_batch],
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

        files = self.get_k_relevant_documents(vector_db_from_all_files, query, k=5)

        relevant_files = self.sort_docs_by_relevance_scores(files, self.cutoff_score)

        if relevant_files is None:
            self.call('showNoRelevantDataMessage')
        else:
            print('Total number of relevant files: ', len(relevant_files))

            sources = [doc.metadata.get('source', None) for doc, _score in relevant_files]

            scores = [round(_score, 2) for doc, _score in relevant_files]

            text_chunks = []

            for file, source in zip(relevant_files, sources):

                text_chunks.extend(self.split_document(file, source))

            vector_db_from_relevant_chunks_only = self.create_vector_db(documents=text_chunks, chroma_path='', persist=False)

            relevant_text_chunks = self.get_k_relevant_documents(vector_db_from_relevant_chunks_only, query, k=3)

            relevant_text_chunks_based_on_criteria = self.sort_docs_by_relevance_scores(relevant_text_chunks, self.cutoff_score)

            if relevant_text_chunks_based_on_criteria is None:

                self.call('showNoRelevantDataMessage')

            else:

                print("Total number of relevant chunks: ", len(relevant_text_chunks_based_on_criteria))

                print("\nRelevant chunk headers:")

                for relevant_text_chunk_based_on_criteria in relevant_text_chunks_based_on_criteria:

                    print(relevant_text_chunk_based_on_criteria[0].metadata['headers'])

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

        best = []

        if scores[0] <= float(cutoff_score):
            return None

        for i in range(len(scores) - 1):

            curr = scores[0]
            next = scores[i + 1]
            best.append(documents[i])
            if ((curr - next) / curr) >= 0.10:
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

    def expand(self, directory_id):
        directory = Directory.objects.filter(id=directory_id).first()
        if directory:
            directory_structure = directory.structure
            print(directory_structure)
            directory_files_status = {}
            for file in File.objects.filter(directory_id=directory_id).all():
                directory_files_status[str(file.file)] = file.processed
            print(directory_files_status)
            self.call('expand', directory_id, json.dumps(directory_structure), json.dumps(directory_files_status))


    def get_first_paragraph_from_text(self, text, type):

        text = "\n".join([line for line in text.strip().split('\n') if line][:3])

        if type == 'title':

            text = text.split('##')[0].split('###')[0]

        elif type == 'header':

            text = text.split('###')[0]

        return text