import routers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

#from apis.chatbot_backend import chatbot_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_DIR =  r"/media/abk/New Disk/DATASETS/clusterdataset" #NOTE:utils.IMG_DIR
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

#app.include_router(chatbot_router) 
routers_list = [
    routers.backend_router,
    routers.chatbot_router, #NOTE
    routers.clustering_router,
    routers.guess_species_router,
    routers.student_router,
    routers.admin_router,
    routers.assitant_router
]
for router in routers_list:
    app.include_router(router=router) 



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

