import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def temporal_split(ratings):
    ordered=ratings.sort_values(["user","timestamp","item"])
    # Leave each user's last interaction for the test set, previous one for validation.
    test=ordered.groupby("user",sort=False).tail(1)
    remaining=ordered.drop(test.index);validation=remaining.groupby("user",sort=False).tail(1)
    train=remaining.drop(validation.index)
    return train,validation,test

def fit(ratings,items):
    ids=sorted(items);item_index={v:i for i,v in enumerate(ids)}
    users=sorted(ratings.user.unique());user_index={v:i for i,v in enumerate(users)}
    positives=ratings[ratings.rating>=4]
    matrix=csr_matrix((np.ones(len(positives)),([user_index[u] for u in positives.user],[item_index[i] for i in positives.item])),shape=(len(users),len(ids)))
    # Normalized item co-occurrence approximates cosine similarity; zero diagonal avoids self-recommendations.
    columns=normalize(matrix.T,norm="l2",axis=1)
    similarity=(columns@columns.T).toarray().astype(np.float32);np.fill_diagonal(similarity,0)
    popularity=np.asarray(matrix.sum(axis=0)).ravel();popularity=popularity/max(popularity.max(),1)
    histories={int(u):[(int(row.item),float(row.rating)) for row in g.itertuples()] for u,g in ratings.groupby("user")}
    return {"ids":ids,"item_index":item_index,"items":items,"similarity":similarity,"popularity":popularity,"histories":histories}

def rank(model,user,events=None,k=10,alpha=.9):
    history={i:r for i,r in model["histories"].get(user,[])}
    history.update({int(i):float(r) for i,r in (events or [])})
    known=[(model["item_index"][i],r) for i,r in history.items() if i in model["item_index"]]
    positive=[(i,r-3) for i,r in known if r>=4]
    scores=model["popularity"].copy().astype(float)
    if positive:
        weights=np.array([r for _,r in positive]);personal=np.average(model["similarity"][[i for i,_ in positive]],axis=0,weights=weights)
        maximum=personal.max();personal=personal/maximum if maximum>0 else personal
        scores=alpha*personal+(1-alpha)*scores
    for i,_ in known:scores[i]=-np.inf
    order=np.argsort(-scores,kind="stable")[:k]
    return [{"item_id":model["ids"][i],"title":model["items"][model["ids"][i]]["title"],"score":float(scores[i]),"reason":"item-similarity + popularity" if positive else "popularity cold-start"} for i in order if np.isfinite(scores[i])]

def ranking_metrics(model,holdout,alpha=.9,k=10):
    # Full-catalog ranking, excluding seen interactions. Only relevant held-out ratings (>=4) are evaluated.
    cases=holdout[holdout.rating>=4];hits=[];ndcg=[]
    for row in cases.itertuples():
        candidates=[x["item_id"] for x in rank(model,int(row.user),k=k,alpha=alpha)]
        hit=row.item in candidates;hits.append(float(hit));ndcg.append(1/np.log2(candidates.index(row.item)+2) if hit else 0)
    return {"users_with_relevant_holdout":len(cases),"recall_at_10":float(np.mean(hits)) if hits else 0,"ndcg_at_10":float(np.mean(ndcg)) if ndcg else 0,"precision_at_10":float(np.mean(hits)/k) if hits else 0}
