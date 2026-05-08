# Agente Funcional DUOC UC — LangGraph + RAG


Agente funcional autonomo basado en LangGraph, responde consultas estudiantiles acerca del Reglamento Academico de Duoc UC mediante el patron ReAct.

Desarrollado como parte de la Evaluación Parcial N°2 del curso
ISY0101 — Ingeniería de Soluciones con IA.

---

## Requisitos previos

- GitHub Codespaces (recomendado) o Python 3.12+
- Variable "GITHUB_TOKEN" configurada. En otro entorno ejecuta: "export GITHUB_TOKEN="tu_token""

---

## Instalación

1. Clona el repositorio:
   git clone https://github.com/wiLk4nn2/Agente-funcional-ia-duoc.git
   cd Agente-funcional-ia-duoc

2. Instala las dependencias:
   pip install -r requerimientos.txt

---

## Cómo ejecutar

1. Abre el archivo "agente-duoc.py"
2. Ejecuta el codigo completo, esto realiza:
   - Carga y vectorización del Reglamento Académico en FAISS
   - Definición de herramientas del agente
   - Configuración de memoria con MemorySaver
   - Construcción del agente ReAct con LangGraph
   - Pruebas automáticas del sistema

4. Para hacer tu propia consulta, modifica cualquiera de las últimas líneas:
   print(consultar_agente("tu pregunta aquí"))

- El agente mantiene memoria dentro de la misma sesión.
- Para iniciar una nueva sesión, cambia el valor de thread_id.

---

## Arquitectura del agente

El agente implementa el patrón ReAct (Reasoning + Acting + Observing)
sobre LangGraph 1.2.17, con los siguientes componentes:

Herramientas:
- buscar_reglamento: Recupera fragmentos del Reglamento Académico
  mediante búsqueda semántica en FAISS y genera respuesta con cita al
  artículo fuente.
- resumir_normativa: Resume y estructura en puntos clave una respuesta
  extensa del reglamento.

Memoria:
- Corto plazo: MemorySaver — mantiene el historial de la sesión
  actual por thread_id.
- Largo plazo: FAISS — base vectorial del Reglamento Académico,
  disponible para todas las sesiones.

---

## Tecnologías utilizadas

- LangGraph 1.2.17 — framework de agentes con patrón ReAct
- FAISS — base de datos vectorial en memoria
- OpenAI text-embedding-3-small — generación de embeddings
- GPT-4.1 — modelo de lenguaje (temperatura 0.1)
- PyPDF — extracción de texto desde PDF

---

## Evidencia de pruebas

Las capturas de las respuestas del agente ante consultas reales
se encuentran en la carpeta /evidencia.

Escenarios probados:
- Consulta normativa directa (Artículo N°42)
- Pregunta de seguimiento usando memoria conversacional
- Solicitud de resumen estructurado
- Consulta fuera del dominio del reglamento

---

## Relación con EP1

Este proyecto extiende el pipeline RAG desarrollado en la
Evaluación Parcial 1. El vector store FAISS y el Reglamento
Académico de DUOC UC son reutilizados directamente como
memoria de largo plazo del agente.
