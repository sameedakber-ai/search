from django_unicorn.components import UnicornView

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


class LlmView(UnicornView):
    question = "How do I do this?"
    conversation = []
    answer = ''

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
        db = self.get_vector_db(chroma_path="media/chroma")
        relevant_docs = self.get_k_relevant_documents(db, self.question, k=1)
        sources = [doc.metadata.get('source', None) for doc, _score in relevant_docs]
        llm_response = self.get_llm_response(self.question, relevant_docs)
        formatted_response = f"Response: {llm_response}\nSources: {sources}"
        print(formatted_response)
        self.answer = formatted_response
        self.conversation.append((self.question, self.answer))







