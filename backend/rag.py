import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from backend.config import (
    GROQ_API_KEY, CHROMA_DB_PATH, CHUNK_SIZE, DOCUMENTS_PATH,
    CHUNK_OVERLAP, TOP_K_RESULTS, MODEL_NAME, EMBEDDING_MODEL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for path in [CHROMA_DB_PATH, DOCUMENTS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

chat_histories = {}
vectorstore = None

# Load embedding model once at startup
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from {file_path}")
    return documents


def process_document(file_path: str):
    global vectorstore

    # Step 1: Load
    documents = load_document(file_path)

    # Step 2: Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    # Step 3: Embed locally + Store in ChromaDB
    if vectorstore is not None:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    logger.info("Document stored in ChromaDB")
    return len(chunks)


def get_answer(question: str, session_id: str):
    global vectorstore, chat_histories

    # Prompt injection guard
    banned = ["ignore previous", "forget instructions", "act as","act like", "jailbreak", "disregard"]
    if any(phrase in question.lower() for phrase in banned):
        return {
            "answer": "This information is not present in the provided document.",
            "sources": [],
            "similarity_scores": [],
            "confidence": "blocked"
        }

    if vectorstore is None:
        return {
            "answer": "No document has been uploaded yet. Please upload a document first.",
            "sources": [],
            "similarity_scores": [],
            "confidence": "no_document"
        }


    # Dynamic retrieval: if document is small/medium, retrieve up to 20 chunks so the LLM understands the whole document
    try:
        all_ids = vectorstore.get().get("ids", [])
        total_chunks = len(all_ids)
    except Exception:
        total_chunks = 100

    target_k = min(total_chunks, 18) if total_chunks <= 25 else TOP_K_RESULTS
    fetch_k = max(target_k * 2, 20)

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": target_k,
            "fetch_k": fetch_k
        }
    )

    retriever_with_scores = vectorstore.similarity_search_with_score(
        question, k=target_k
    )

    score_map = {}
    for doc, score in retriever_with_scores:
        score_map[doc.page_content[:50]] = round(float(score), 4)

    mmr_docs = mmr_retriever.invoke(question)

    sources = []
    similarity_scores = []
    for doc in mmr_docs:
        key = doc.page_content[:50]
        score = score_map.get(key, 0.0)
        sources.append({
            "content": doc.page_content[:300],
            "page": doc.metadata.get("page", "unknown")
        })
        similarity_scores.append(score)

    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in mmr_docs
    ])

    if session_id not in chat_histories:
        chat_histories[session_id] = []
    history = chat_histories[session_id]

    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.2
    )

    # Intelligent comprehension prompt: handles direct facts, indirect reasoning, and word meanings
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent, highly perceptive document analysis assistant.
Your mission is to understand the provided document deeply and answer both direct and indirect questions based on it.

How to answer:
1. Holistic Comprehension: Synthesize the document as a whole. Understand overall themes, purposes, goals, and context, not just literal keyword matches.
2. Indirect & Conceptual Questions: When asked analytical, conceptual, or indirect questions (e.g., "Why was this created?", "What is the purpose?", "What are the benefits or implications?", "Summarize the key points"), infer logical, well-reasoned conclusions that are firmly grounded in the document content.
3. Explaining Meanings & Vocabulary: When asked about the meaning or definition of specific words, phrases, jargon, or acronyms used in the document, explain what they mean in general and specifically how they apply within the context of the document.
4. Factual & Faithful: Your answers must be faithful to the document. Do not invent unrelated claims or contradict the document.
5. Not in Document: Only if a question is completely unrelated to the document and cannot be answered or inferred from the provided context, respond politely with:
   "This information is not mentioned or discussed in the provided document."
6. Citations: Mention the relevant page number(s) or sections whenever applicable.

Document Context:
{context}

Chat History:
{history}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "history": history_text,
        "question": question
    })

    # Confidence score based on best similarity score
    best_score = min(similarity_scores) if similarity_scores else 999
    if best_score < 0.5:
        confidence = "very_high"
    elif best_score < 1.0:
        confidence = "high"
    elif best_score < 1.5:
        confidence = "medium"
    else:
        confidence = "low"


    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    logger.info(f"Question: {question} | Session: {session_id} | Confidence: {confidence}")

    return {
        "answer": answer,
        "sources": sources,
        "similarity_scores": similarity_scores,
        "confidence": confidence
    }