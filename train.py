import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from app.ranking import fit, temporal_split, ranking_metrics

def load_data():
    ratings=pd.read_csv("data/ml-100k/u.data",sep="\t",names=["user","item","rating","timestamp"])
    raw=pd.read_csv("data/ml-100k/u.item",sep="|",encoding="latin-1",header=None)
    items={int(row[0]):{"title":row[1]} for row in raw.itertuples(index=False,name=None)}
    return ratings,items

def train():
    ratings,items=load_data();train,validation,test=temporal_split(ratings)
    model=fit(train,items)
    validation_metrics={str(alpha):ranking_metrics(model,validation,alpha) for alpha in [.5,.75,.9]}
    alpha=float(max(validation_metrics,key=lambda a:validation_metrics[a]["ndcg_at_10"]))
    # Refit with validation interactions after selecting blend weight; test remains untouched.
    final=fit(pd.concat([train,validation]),items)
    measured=ranking_metrics(final,test,alpha)
    baseline=ranking_metrics(final,test,alpha=0)
    report={"dataset":"MovieLens 100K", "ratings":len(ratings),"users":ratings.user.nunique(),"items":len(items),"split":{"train":len(train),"validation":len(validation),"test":len(test)},"protocol":"Per-user last interaction test, penultimate validation; full-catalog ranking excludes previously seen items. Relevance: rating >=4. Other users may have later interactions: this is per-user temporal, not global temporal evaluation.","selected_alpha":alpha,"validation":validation_metrics,"test":measured,"popularity_baseline":baseline}
    final["alpha"]=alpha;final["version"]=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()[:12];final["report"]=report
    Path("artifacts").mkdir(exist_ok=True);joblib.dump(final,"artifacts/recommender.joblib")
    Path("reports").mkdir(exist_ok=True);Path("reports/evaluation.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return final
if __name__=="__main__":train()
