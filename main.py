from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain import hub
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline

DATA_DIR = "data/"
EMBEDDING_MODEL = "microsoft/Phi-3-mini-4k-instruct"
GENERATION_MODEL = "microsoft/Phi-3-mini-4k-instruct"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

CONTEXT:
{context}

Question: {question}

Helpful Answer:"""


def make_vector_store():
    print("Loading raw articles")
    documents = DirectoryLoader(DATA_DIR).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    print("Loading embedding model")
    model_kwargs = {"device": "cpu"}
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
    )

    print("Creating embeddings")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    del embedding_model
    return vector_store


def init_retriever(k=6):
    print("Making vector store")
    if os.path.isfile("vec_store.pkl"):
        print("Found saved vector store")
        vector_store = pickle.load(open("vec_store.pkl", "rb"))
    else:
        print("Computing and saving vector store")
        vector_store = make_vector_store()
        pickle.dump(vector_store, open("vec_store.pkl", "wb"))

    print("Creating retriever")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    retriever = init_retriever()
    questions = [
        "What did the OpenAI model do?",
        "What is happening at McDonalds?",
        "Why is the MCDonald's outbreak important?",
        "What can be done to mitigate the OpenAI issue?",
    ]

    llm = HuggingFacePipeline.from_model_id(
        model_id=GENERATION_MODEL,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 75,
            "do_sample": False,
        },
    )
    prompt_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | PromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = prompt_chain | llm
    for question in questions:
        prompt = prompt_chain.invoke(question)
        answer = llm.invoke(prompt).replace(prompt.text + "\n\n", "")
        print("=" * 10)
        print(f"Question: {question}\n\nAnswer: {answer}")
        print("=" * 10)

    print("DONE!")


if __name__ == "__main__":
    main()
