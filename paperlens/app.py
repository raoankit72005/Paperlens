import json,time
from pathlib import Path
import streamlit as st
from .engine import SearchEngine
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='PaperLens',page_icon='📚',layout='wide')
st.title('PaperLens');st.caption('Research paper search · BM25, semantic retrieval and reciprocal rank fusion')
backend=st.sidebar.selectbox('Embedding backend',['transformer','lsa','bm25']);mode=st.sidebar.selectbox('Ranking',['bm25'] if backend=='bm25' else ['hybrid','bm25','semantic']);alpha=st.sidebar.slider('Semantic fusion weight',0.,1.,.5);rerank=st.sidebar.checkbox('Cross-encoder reranking', value=True)
candidate_k=st.sidebar.selectbox('Candidate pool',[20,50,100])
upload=st.sidebar.file_uploader('Use your corpus (.json)',type=['json'])
@st.cache_resource
def load_engine(raw,backend): return SearchEngine(json.loads(raw),backend)
try:
    engine=load_engine(upload.getvalue().decode() if upload else (ROOT/'data/papers.json').read_text(),backend)
except Exception as e:
    st.error(f'Could not load corpus/model: {e}');st.stop()
if backend=='lsa': st.info('Offline LSA baseline. Select transformer for pretrained sentence embeddings. The bundled corpus contains curated summaries, not full papers.')
q=st.text_input('Search papers',value='fine tune a language model with limited GPU memory')
topic=st.sidebar.selectbox('Topic',['All']+sorted({p['topic'] for p in engine.papers}));years=[p.get('year',0) for p in engine.papers if p.get('year',0)>0]
year=None
if years and min(years)<max(years):
    year=st.sidebar.slider('Published from',min(years),max(years),min(years))
if q:
    try:
        start=time.perf_counter();results=engine.search(q,mode,alpha=alpha,topic=None if topic=='All' else topic,min_year=year,rerank=rerank,candidate_k=candidate_k);st.caption(f'{len(results)} results · {(time.perf_counter()-start)*1000:.1f} ms')
        if not results: st.info('No matching papers. Try another query or remove filters.')
        for rank,p in enumerate(results,1):
            st.subheader(f"{rank}. {p['title']}");st.caption(f"{p['year']} · {p['topic']}");st.write(p['summary']);
            if p.get('url'): st.link_button('Read paper',p['url'])
            with st.expander('Retrieval scores'): st.json({k:p[k] for k in ['score','retrieval_score','bm25','semantic','rerank_score'] if k in p})
    except Exception as e: st.error(f'Search failed: {e}')
