import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import Depends,FastAPI,Header,HTTPException,Query
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from .events import EventStore
from .ranking import rank

def authorize(x_api_key:str|None=Header(default=None)):
    key=os.getenv("API_KEY","")
    if key:
        import hmac
        if not x_api_key or not hmac.compare_digest(key,x_api_key):raise HTTPException(401,"Invalid API key")
class Event(BaseModel):
    user_id:int=Field(ge=1)
    item_id:int=Field(ge=1)
    rating:float=Field(ge=1,le=5)
def create_app(model_path=None,event_db=None):
    @asynccontextmanager
    async def lifespan(app):
        p=Path(model_path or os.getenv("MODEL_PATH","artifacts/recommender.joblib"));app.state.model=joblib.load(p) if p.exists() else None
        app.state.events=EventStore(event_db or os.getenv("EVENT_DB","data/events.db"));yield
    app=FastAPI(title="Real-Time Recommendations",lifespan=lifespan)
    def model():
        if app.state.model is None:raise HTTPException(503,"Run python download_data.py && python train.py first")
        return app.state.model
    @app.get("/")
    def index():return FileResponse(Path(__file__).parent/"web.html")
    @app.get("/health")
    def health():return {"ready":app.state.model is not None}
    @app.get("/catalog",dependencies=[Depends(authorize)])
    def catalog(q:str=Query(default="",max_length=100),limit:int=Query(default=20,ge=1,le=100)):
        return [{"item_id":i,**v} for i,v in model()["items"].items() if q.lower() in v["title"].lower()][:limit]
    @app.post("/events",dependencies=[Depends(authorize)])
    def event(e:Event):
        if e.item_id not in model()["item_index"]:raise HTTPException(404,"Movie not in catalog")
        app.state.events.record(e.user_id,e.item_id,e.rating);return {"recorded":True,"visible_on_next_request":True}
    @app.get("/recommendations/{user_id}",dependencies=[Depends(authorize)])
    def recommend(user_id:int,k:int=Query(default=10,ge=1,le=50)):
        if user_id<1:raise HTTPException(422,"User ID must be positive")
        m=model();t=time.perf_counter();events=app.state.events.history(user_id);results=rank(m,user_id,events,k,m["alpha"])
        return {"user_id":user_id,"model_version":m["version"],"feedback_items":len(events),"latency_ms":(time.perf_counter()-t)*1000,"recommendations":results}
    @app.get("/evaluation",dependencies=[Depends(authorize)])
    def evaluation():return model()["report"]
    return app
app=create_app()
