"""BM25, LSA / sentence-transformer retrieval, RRF and optional reranking."""
import re, math, collections, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

class SearchEngine:
    def __init__(self, papers, backend='lsa', model='sentence-transformers/all-MiniLM-L6-v2'):
        if not papers or len({p['id'] for p in papers}) != len(papers):
            raise ValueError('Provide a nonempty corpus with unique paper IDs')
        self.papers = papers
        self.texts = [p['title']+' '+p['summary'] for p in papers]
        self.tokens = [tokenize(t) for t in self.texts]
        self.counts = [collections.Counter(t) for t in self.tokens]
        self.lengths = np.array([len(t) for t in self.tokens])
        self.avg_length = max(float(self.lengths.mean()), 1)
        df = collections.Counter(w for t in self.tokens for w in set(t))
        self.idf = {w: math.log(1+(len(papers)-n+.5)/(n+.5)) for w,n in df.items()}
        self.backend = backend
        self.encoder = None
        self.reranker = None
        if backend == 'transformer':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model, device='cpu')
            self.vectors = self.encoder.encode(self.texts, normalize_embeddings=True)
        elif backend == 'lsa':
            self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[a-z0-9]+\b')
            x = self.vectorizer.fit_transform(self.texts)
            self.svd = TruncatedSVD(n_components=min(16,x.shape[0]-1,x.shape[1]-1),random_state=42)
            self.vectors = normalize(self.svd.fit_transform(x))
        else:
            raise ValueError('backend must be lsa or transformer')

    def bm25(self, query):
        score = np.zeros(len(self.papers))
        for word in set(tokenize(query)):
            tf = np.array([c[word] for c in self.counts])
            score += self.idf.get(word,0)*tf*2.5/(tf+1.5*(.25+.75*self.lengths/self.avg_length))
        return score

    def semantic(self, query):
        if self.encoder is not None:
            q = self.encoder.encode([query],normalize_embeddings=True)
        else:
            q = normalize(self.svd.transform(self.vectorizer.transform([query])))
        return (self.vectors @ q.T).ravel()

    def search(self, query, mode='hybrid', k=10, alpha=.5, topic=None, min_year=None, rerank=False):
        if mode not in ('bm25','semantic','hybrid') or not 0<=alpha<=1 or k<1:
            raise ValueError('Invalid search configuration')
        if not query.strip():
            return []
        eligible = np.array([i for i,p in enumerate(self.papers) if (not topic or p['topic']==topic) and (not min_year or p['year']>=min_year)],dtype=int)
        if not len(eligible):
            return []
        b,s=self.bm25(query),self.semantic(query)
        br=sorted(eligible,key=lambda i:(-b[i],self.papers[i]['id']))
        sr=sorted(eligible,key=lambda i:(-s[i],self.papers[i]['id']))
        fusion=np.zeros(len(self.papers))
        # Omit unmatched terms/vectors instead of giving irrelevant documents arbitrary rank credit.
        for weight,ranking,scores in ((1-alpha,br,b),(alpha,sr,s)):
            for rank,i in enumerate([i for i in ranking if scores[i]>1e-10],1):
                fusion[i]+=weight/(60+rank)
        scores={'bm25':b,'semantic':s,'hybrid':fusion}[mode]
        ranking=sorted([i for i in eligible if scores[i]>1e-10],key=lambda i:(-scores[i],self.papers[i]['id']))
        cross={}
        if rerank and ranking:
            from sentence_transformers import CrossEncoder
            if self.reranker is None:
                self.reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',device='cpu')
            candidates=ranking[:50]
            cross=dict(zip(candidates,map(float,self.reranker.predict([(query,self.texts[i]) for i in candidates]))))
            ranking=sorted(candidates,key=lambda i:-cross[i])
        return [dict(self.papers[i],score=float(scores[i]),bm25=float(b[i]),semantic=float(s[i]),**({'rerank_score':cross[i]} if i in cross else {})) for i in ranking[:k]]
