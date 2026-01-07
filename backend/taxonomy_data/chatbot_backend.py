from enum import Enum
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.schema import SystemMessage, HumanMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse

# ---------- CONFIG ----------
DB_FAISS_PATH = r"/home/abk/abk/projects/Major-project-basic-ui/backend/vectorstore/"
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
CHAT_TYPE_DETECTION = r"/home/abk/abk/projects/Major-project-basic-ui/backend/vectorstore/chat_type_detection_embed.npz"


#------------ FUNCTIONS ------------------
class TaxonomyChatType(Enum):
    SMALL_TALK = "small_talk"
    TAXONOMY_FROM_SPECIES = "taxonomy_from_species"
    TAXONOMY_FROM_FEATURES = "taxonomy_from_features"

def Load_ChatType_Embed():
    with np.load(CHAT_TYPE_DETECTION) as data:
        return {TaxonomyChatType(intent): data[intent] for intent in data.files}

def get_ChatType(query):
    query_vec = np.array(embeddings.embed_query(query)).reshape(1, -1)
    best_intent = "taxonomy_from_species" # Default fallback
    highest_score = 0
    
    for intent, vectors in test_vectors.items():
        scores = cosine_similarity(query_vec, vectors)
        max_score = np.max(scores)
        if max_score > highest_score:
            highest_score = max_score
            best_intent = intent
    # If the match is very weak (under 0.4), treat it as a general query
    return best_intent if highest_score > 0.4 else "taxonomy_from_species"

# -----------------------------------

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# Load vectorstore
vectorstore = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Load chat model (STREAMING ENABLED)
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
    disable_streaming=False,
    seed=43,
    #num_predict=100
)

test_vectors =  Load_ChatType_Embed()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------

@app.post("/taxonomyChat")
async def taxonomyChat(user_input: str = Form(...), files: List[UploadFile] = File([]) ):
    
    #TODO:thampuran ariya
    if files:  
        return {"error": "Not implemented"} 
    
    intent = get_ChatType(user_input)

    if intent == TaxonomyChatType.SMALL_TALK:
        messages = [
            SystemMessage(
                "You are an AI assistant designed for Marine Taxonomists specialist in Glyceridae, Polychaeta"
                "Assist users only related to these topics. Small comverstational chats and basic greetings are fine"
                "Never talk about thing from your own knowledge. always stick with context"
                "You are owned by CMLRE which stands for The Centre for Marine Living Resources & Ecology is a research institute in Kochi, Kerala under the Ministry of Earth Sciences, Government of India with a mandate to study the marine living resources"
                "Your name is TAXOBOT"
                "Keep answers minimal"

            ),
            HumanMessage(
                f"user: {user_input}\n"
                "chatbot:"
            )
        ]
        
    elif intent == TaxonomyChatType.TAXONOMY_FROM_SPECIES:
        print(TaxonomyChatType.TAXONOMY_FROM_SPECIES)
        docs_and_scores = vectorstore.similarity_search_with_score(user_input, k=3)
        if docs_and_scores[0][1] > 0.8: 
            return PlainTextResponse("I cannot identify this specimen from the provided data.")
            
        context = "\n\n".join(
            f"[Doc {i+1}]\n{d[0].page_content}"
            for i, d in enumerate(docs_and_scores)
        )

        messages = [
            SystemMessage(
                "You are an expert Marine Taxonomist. You should explain about species when their name is given"
                "You MUST answer using ONLY the provided context. "
                "If the answer is not present, reply exactly:\n"
                "'I cannot identify this specimen from the provided data.'"
                "Dont specify from which doc it is specified or how you found it. Just tell the answer in a pleasent humanly way like taking to someone.But keep it short"
            ),
            HumanMessage(
                f"Context:\n{context}\n\nQuestion:\n{user_input}\nChatbot:"
            )
        ]

    elif intent == TaxonomyChatType.TAXONOMY_FROM_FEATURES:
        print(TaxonomyChatType.TAXONOMY_FROM_FEATURES)
        docs_and_scores = vectorstore.similarity_search_with_score(user_input, k=12)
        if docs_and_scores[0][1] > 0.8: 
            return PlainTextResponse("I cannot identify this specimen from the provided data.")
            
        context = "\n\n".join(
            f"[Doc {i+1}]\n{d[0].page_content}"
            for i, d in enumerate(docs_and_scores)
        )

        messages = [
            SystemMessage(
                "You are an expert Marine Taxonomist. You should explain about species when their name is given"
                "You MUST answer using ONLY the provided context. "
                "If the answer is not present, reply exactly:\n"
                "'I cannot identify this specimen from the provided data.'"
                "Dont specify from which doc it is specified or how you found it. Just tell the answer in a pleasent humanly way like taking to someone.But keep it short"
            ),
            HumanMessage(
                f"Context:\n{context}\n\nQuestion:\n{user_input}"
            )
        ]


    async def event_stream():
        full_response = ""
        async for chunk in llm.astream(messages):
            token = chunk.text
            full_response += token
            yield token

    return StreamingResponse(event_stream(), media_type="text/plain")


# =========================================================
# Run server
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
