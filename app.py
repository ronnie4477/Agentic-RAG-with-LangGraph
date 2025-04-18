import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Agentic AI Retriever", layout="centered")
st.title("🔍 Agentic AI Chatbot")
st.markdown("Enter multiple URLs below and ask a question based on the content.")

# URL Input
url_input = st.text_area("Enter URLs (one per line):", height=150)
question = st.text_input("Ask a question related to the URLs content")

if st.button("Submit and Ask"):
    with st.spinner("Loading and processing..."):

        urls = [url.strip() for url in url_input.split("\n") if url.strip()]
        if not urls or not question:
            st.warning("Please provide at least one URL and a question.")
            st.stop()

        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)

        # Create vector store
        vectorstore = FAISS.from_documents(doc_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        retriever_tool = create_retriever_tool(retriever, "retriever_vector_db_blog", "Search and run information about Agentic AI")
        tools = [retriever_tool]

        # State definition
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # Nodes
        def agent(state):
            messages = state["messages"]
            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7).bind_tools(tools)
            response = model.invoke(messages)
            return {"messages": [response]}

        def grade_documents(state) -> Literal["generate", "rewrite"]:
            class Grade(BaseModel):
                binary_score: str = Field(description="Relevance score 'yes' or 'no'")

            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            chain = (
                PromptTemplate(
                    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                    Here is the retrieved document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
                    input_variables=["context", "question"],
                )
                | model.with_structured_output(Grade)
            )

            question = state["messages"][0].content
            docs = state["messages"][-1].content
            result = chain.invoke({"question": question, "context": docs})
            return "generate" if result.binary_score == "yes" else "rewrite"

        def generate(state):
            messages = state["messages"]
            question = messages[0].content
            docs = messages[-1].content
            prompt = hub.pull("rlm/rag-prompt")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            rag_chain = prompt | llm | StrOutputParser()
            response = rag_chain.invoke({"context": docs, "question": question})
            return {"messages": [AIMessage(content=response)]}

        def rewrite(state):
            question = state["messages"][0].content
            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            response = model.invoke([
                HumanMessage(
                    content=f""" \n 
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                    Here is the initial question:
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    Formulate an improved question: """
                )
            ])
            return {"messages": [response]}

        # Graph definition
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent)
        retrieve = ToolNode([retriever_tool])
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("rewrite", rewrite)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
        workflow.add_conditional_edges("retrieve", grade_documents)
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        graph = workflow.compile()

        result = graph.invoke({"messages": [HumanMessage(content=question)]})

        # Final output
        final_message = result["messages"][-1].content
        st.success("Answer:")
        st.markdown(final_message)
