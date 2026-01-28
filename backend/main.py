import routers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#from apis.chatbot_backend import chatbot_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.include_router(chatbot_router) 
routers_list = [
    routers.backend_router,
    routers.clustering_router,
    routers.chatbot_router, #NOTE
]
for router in routers_list:
    app.include_router(router=router) 



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

