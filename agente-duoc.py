from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
import os

os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFLoader("docs/reglamento-academico-duoc-2024.pdf")
documents = loader.load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(documents)
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

@tool
def buscar_reglamento(consulta: str) -> str:
    """Busca información en el Reglamento Académico oficial de DUOC UC.
    Usar cuando el estudiante pregunta sobre normativas, inasistencias,
    reprobación, titulación, becas o cualquier procedimiento institucional."""
    docs = retriever.invoke(consulta)
    contexto = "\n\n".join(doc.page_content for doc in docs)
    response = llm.invoke(f"""Eres un asistente académico de DUOC UC.
Responde SOLO basándote en el siguiente contexto del Reglamento Académico.
Cita siempre el artículo fuente. Si no encuentras la respuesta, indícalo.
No solicites ni menciones datos personales del estudiante.

Contexto:
{contexto}

Consulta: {consulta}""")
    return response.content

@tool
def resumir_normativa(texto: str) -> str:
    """Resume y estructura en puntos clave una normativa o respuesta extensa.
    Usar cuando la respuesta necesita ser simplificada para el estudiante."""
    response = llm.invoke(f"""Resume el siguiente texto en máximo 5 puntos clave,
usando lenguaje claro para un estudiante de DUOC UC.
Mantén referencias a artículos si los hay.

Texto:
{texto}""")
    return response.content

tools = [buscar_reglamento, resumir_normativa]

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "sesion_duoc_1"}}

def consultar_agente(consulta: str) -> str:
    response = agent.invoke(
        {"messages": [("user", consulta)]},
        config=config
    )
    return response["messages"][-1].content

if __name__ == "__main__":
    print("=== PRUEBA 1: Consulta directa ===")
    print(consultar_agente("¿Qué pasa si repruebo una asignatura por tercera vez?"))

    print("\n=== PRUEBA 2: Seguimiento con memoria ===")
    print(consultar_agente("¿Y si es la segunda vez que la repruebo?"))

    print("\n=== PRUEBA 3: Solicitud de resumen ===")
    print(consultar_agente("Resume eso en puntos clave"))

    print("\n=== PRUEBA 4: Fuera del alcance ===")
    print(consultar_agente("¿Por que no tienen permitido vender helados despues de las 18pm en el Duoc?"))