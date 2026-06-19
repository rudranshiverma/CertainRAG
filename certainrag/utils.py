import numpy as np

#the function below converts L2 distances to similarity scores in [0,1]
#use if your retriever (such as default FAISS) returns raw L2 distance instead of normalized cosine sim.

def normalize_l2(distances:list[float])->list[float]:
    arr=np.array(distances,dtype=float)
    similarities=1.0/(1.0+arr)
    return similarities.tolist()

"""example:
distances=[d for _, d in vectorstore.similarity_search_with_score(query)]
scores=normalize_l2(distances)"""