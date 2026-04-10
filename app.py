import streamlit as st
import os
import json
import voyageai
import plotly.graph_objects as go

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent

from langgraph.graph import StateGraph

st.set_page_config(page_title="AI PDF Tool", layout="wide")

groq_key = st.secrets["GROQ_API_KEY"]
voy_key = st.secrets["VOY_API_KEY"]

if not groq_key or not voy_key:
    st.error("Keys missing")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_key,
    model="llama3-8b-8192"
)

voy_client = voyageai.Client(api_key=voy_key)

class VoyageEmbed:
    def embed_documents(self, txts):
        return voy_client.embed(txts, model="voyage-2").embeddings

    def embed_query(self, txt):
        return voy_client.embed([txt], model="voyage-2").embeddings[0]

st.title("PDF AI helper")

file = st.file_uploader("upload pdf", type="pdf")
q = st.text_input("ask anything")

@st.cache_resource
def make_db(pth):
    loader = PyPDFLoader(pth)
    docs = loader.load()

    split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    parts = split.split_documents(docs)

    emb = VoyageEmbed()

    db = FAISS.from_documents(parts, emb)

    return db.as_retriever()

def make_chart(js):
    kws = js.get("keywords", []) or ["none"]

    fig = go.Figure(data=[
        go.Bar(x=kws, y=[1]*len(kws))
    ])

    fig.update_layout(
        title=js.get("title", "chart")
    )

    return fig

if file:
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", file.name)

    with open(path, "wb") as f:
        f.write(file.read())

    with st.spinner("processing..."):
        retr = make_db(path)

    def sum_tool(t):
        return llm.invoke(f"Summarize this:\n{t}").content

    def ques_tool(t):
        return llm.invoke(f"Make 5 MCQs from:\n{t}").content

    def js_tool(t):
        return llm.invoke(f"""
        Return ONLY JSON. No extra text.

        {{
          "title": "",
          "summary": "",
          "keywords": []
        }}

        Text:
        {t}
        """).content

    def html_maker(t):
        return llm.invoke(f"""
        Make simple HTML infographic page from this JSON.
        Show title, summary and keywords nicely.

        JSON:
        {t}
        """).content

    tools = [
        Tool(name="summ", func=sum_tool, description="summary"),
        Tool(name="mcq", func=ques_tool, description="questions"),
        Tool(name="json", func=js_tool, description="json output"),
        Tool(name="html", func=html_maker, description="html page")
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False
    )

    class State(dict):
        pass

    def step1(s):
        return {"query": s["query"]}

    def step2(s):
        d = retr.get_relevant_documents(s["query"])
        ctx = "\n".join([x.page_content for x in d])
        return {"context": ctx, "query": s["query"]}

    def step3(s):
        out = agent.run(f"{s['context']}\n{s['query']}")

        try:
            js = json.loads(out)
            return {"answer": out, "json": js}
        except:
            return {"answer": out}

    g = StateGraph(State)

    g.add_node("a", step1)
    g.add_node("b", step2)
    g.add_node("c", step3)

    g.set_entry_point("a")

    g.add_edge("a", "b")
    g.add_edge("b", "c")

    app = g.compile()

    if q:
        with st.spinner("thinking..."):
            res = app.invoke({"query": q})

        st.subheader("output")
        st.write(res.get("answer", "no answer"))

        if "json" in res:
            st.subheader("chart")

            fig = make_chart(res["json"])
            st.plotly_chart(fig)

            html_out = html_maker(json.dumps(res["json"]))

            st.subheader("html")
            st.code(html_out, language="html")
