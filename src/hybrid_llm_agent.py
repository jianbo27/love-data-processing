import os
import json
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# === LOAD ENV ===
load_dotenv()

# === CONFIG ===
SQLITE_PATH = "data/california_schools/california_schools.sqlite"
QDRANT_COLLECTION = "school_vectors"

# === EMBEDDINGS ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === DATABASE SCHEMA INFO ===
def get_db_schema_info() -> str:
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        
        # Get table list
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        
        schema_info = []
        for table in tables:
            table_name = table[0]
            cur.execute(f"PRAGMA table_info({table_name});")
            columns = cur.fetchall()
            column_names = [col[1] for col in columns]
            schema_info.append(f"Table: {table_name}\nColumns: {', '.join(column_names)}\n")
        
        cur.close()
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error getting schema: {e}"

# === TOOL: SQLITE ===
def query_sqlite(sql_query: str) -> str:
    if sql_query.lower().strip() == "help" or sql_query.lower().strip() == "schema":
        return get_db_schema_info()
        
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return json.dumps(rows, indent=2)
    except Exception as e:
        return f"SQLite Error: {e}"

# === TOOL: QDRANT ===
qdrant = QdrantClient("localhost", port=6333)

def query_qdrant(text: str, top_k: int = 5) -> str:
    try:
        vector = embedding_model.encode(text).tolist()
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k
        )
        return json.dumps([hit.payload for hit in hits], indent=2)
    except Exception as e:
        return f"Qdrant Error: {e}"

# === POPULATE QDRANT ===
def populate_qdrant_from_csv(csv_path: str, collection_name: str):
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient='records')
    texts = [json.dumps(record) for record in records]
    vectors = embedding_model.encode(texts).tolist()

    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
    )

    points = [PointStruct(id=i, vector=vectors[i], payload=records[i]) for i in range(len(records))]
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"✅ Loaded {len(records)} records into Qdrant collection '{collection_name}'")

# === TOOLS ===
tools = [
    Tool(name="SQLiteTool", func=query_sqlite, description="Executes SQL queries on a SQLite DB."),
    Tool(name="QdrantTool", func=query_qdrant, description="Performs semantic search on Qdrant vector DB.")
]

# === LLM ===
llm = Ollama(
    model="llama3.1:8b",
    temperature=0.2,
    num_ctx=4096
)

# === PROMPT TEMPLATE ===
react_template = """You are an agent that uses tools to answer questions about California schools.

AVAILABLE DATABASE SCHEMA:
{schema_info}

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: what you should do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Final Answer: the answer to the original question

Begin!

Question: {input}
{agent_scratchpad}"""

def setup_agent():
    schema_info = get_db_schema_info()
    
    react_prompt = PromptTemplate(
        template=react_template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"schema_info": schema_info}  
    )
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    
    # Add error handling
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True  # Add this to handle parsing errors
    )
    
    return agent_executor

# === MAIN ===
if __name__ == "__main__":
    populate_qdrant_from_csv("data/california_schools/database_description/schools.csv", "schools_collection")
    populate_qdrant_from_csv("data/california_schools/database_description/frpm.csv", "frpm_collection")
    populate_qdrant_from_csv("data/california_schools/database_description/satscores.csv", "satscores_collection")

    agent_executor = setup_agent()

    while True:
        try:
            query = input("\nEnter query (or 'exit'): ").strip()
            if query.lower() == "exit":
                break
            result = agent_executor.invoke({"input": query})
            print(f"\n\U0001F50E Final Answer:\n{result['output']}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
