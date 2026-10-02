import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.ranking import fit,rank,temporal_split
from app.events import EventStore
from app.main import create_app

def fixture_model():
    frame=pd.DataFrame([(1,1,5,1),(1,2,5,2),(2,1,5,1),(2,2,5,2),(3,3,5,1),(3,4,5,2)],columns=["user","item","rating","timestamp"])
    m=fit(frame,{i:{"title":f"Movie {i}"} for i in range(1,6)});m.update(alpha=.9,version="test",report={});return m

def test_temporal_split():
    frame=pd.DataFrame([(u,i,5,i) for u in [1,2] for i in range(1,5)],columns=["user","item","rating","timestamp"])
    train,val,test=temporal_split(frame)
    assert not set(train.index)&set(val.index) and not set(train.index)&set(test.index)
    for user in [1,2]:assert train[train.user==user].timestamp.max()<val[val.user==user].timestamp.min()<test[test.user==user].timestamp.min()

def test_events_personalize_immediately_and_exclude_seen():
    m=fixture_model();r=rank(m,999,events=[(3,5)],k=3)
    assert r[0]["item_id"]==4
    assert 3 not in [x["item_id"] for x in r]
    assert all(x["item_id"] not in [1,2] for x in rank(m,1,k=5))

def test_persistence_and_upsert(tmp_path):
    path=str(tmp_path/"e.db");s=EventStore(path);s.record(9,1,5);s.record(9,1,2)
    assert EventStore(path).history(9)==[(1,2.)]

def test_api_validation_and_feedback(tmp_path,monkeypatch):
    p=tmp_path/"m.joblib";joblib.dump(fixture_model(),p);monkeypatch.setenv("API_KEY","test")
    with TestClient(create_app(str(p),str(tmp_path/"e.db"))) as c:
        headers={"X-API-Key":"test"}
        assert c.get("/catalog").status_code==401
        assert c.post("/events",headers=headers,json={"user_id":999,"item_id":99,"rating":5}).status_code==404
        assert c.post("/events",headers=headers,json={"user_id":999,"item_id":3,"rating":8}).status_code==422
        assert c.post("/events",headers=headers,json={"user_id":999,"item_id":3,"rating":5}).status_code==200
        r=c.get("/recommendations/999",headers=headers).json()
        assert r["feedback_items"]==1 and r["recommendations"][0]["item_id"]==4
