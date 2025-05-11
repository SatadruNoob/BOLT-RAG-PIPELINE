import os
import time
from typing import List, Literal, Dict, TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain import hub

# Initialize the Mistral chat model
def setup_llm():
    """Initialize the LLM for RAG pipeline."""
    from langchain_mistralai import ChatMistralAI
    
    if not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    return ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.environ.get("MISTRAL_API_KEY")
    )

# Define the RAG pipeline components
class Search(TypedDict):
    """Search query."""
    query: str
    section: Literal["all_sections"]

class State(TypedDict):
    """State for RAG pipeline."""
    question: str
    query: Search
    context: List[Document]
    answer: str

def setup_rag_pipeline(client, persist_dir, embeddings):
    """
    Set up the RAG pipeline.
    
    Args:
        client: Chroma client
        persist_dir: Directory where Chroma is persisted
        embeddings: Embedding function
        
    Returns:
        Configured RAG pipeline
    """
    from vector_store import unified_search
    from langgraph.graph import StateGraph, START
    
    # Initialize LLM
    llm = setup_llm()
    
    # Get the RAG prompt from LangChain Hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # Define RAG functions
    def analyze_query(state: State) -> Dict:
        """Analyze the question to create a structured query."""
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        return {"query": query}

    def retrieve(state: State) -> Dict:
        """Retrieve relevant documents based on the query."""
        query = state["query"]
        retrieved_docs = unified_search(
            client,
            persist_dir,
            embeddings,
            query=query["query"],
            k=4,
            filter_section=query["section"]
        )
        return {"context": retrieved_docs}

    def generate(state: State) -> Dict:
        """Generate an answer based on the retrieved context."""
        time.sleep(1)  # Rate limit protection
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Build the execution graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    # Define the graph edges
    graph_builder.add_edge(START, "analyze_query")
    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Compile the graph
    graph = graph_builder.compile()
    
    return graph

def ask_question(question: str) -> str:
    """
    Ask a question using the RAG pipeline.
    
    This is a wrapper function that will be called by the Streamlit app.
    In a real implementation, this would use the RAG pipeline initialized in setup_rag_pipeline.
    
    Args:
        question: The question to ask
        
    Returns:
        The answer from the RAG pipeline
    """
    # This is just a placeholder for the Streamlit app
    # In the app, we'll use the actual RAG pipeline from session state
    
    # Create a simple response if no pipeline is set up
    if not os.environ.get("MISTRAL_API_KEY"):
        return "Please set your MISTRAL_API_KEY in the settings panel to enable question answering."
    
    try:
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(model="mistral-large-latest")
        return llm.invoke(f"Answer this question as if you had documents about it: {question}")
    except Exception as e:
        return f"Error processing question: {str(e)}"