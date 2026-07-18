<h1 align="center">🤖 WebPageReader RAG-QnA Bot with LlamaIndex, Gemini </h1>

[Now live on Streamlit Cloud!](https://router-based-rag-bot.streamlit.app)

https://github.com/user-attachments/assets/b8237b02-c397-4520-a478-55429716b6f2

<h5 align="center">Keywords: Python | LlamaIndex | RAG | LLM | Gemini-Pro | NLP | API | Streamlit</h5>
<h2>Project Description</h2>
Developed the simplest form of RAG (Retrieval-Augmented-Generation) with a router. 
RAG is a framework designed capable of tool use, reasoning and decision-making with given data.
Given a query, the router will pick one of the two query engines, QnA or Summarization, to execute a response over a single document, in this case, a website.
<h2>Key Features</h2>
<h3>⚒️ Natural Language Processing</h3>
<ul>
 <li> Utilizes Google's Gemini-Pro large language model to interpret natural language queries. </li>
 <li> The bot was able to do some reasoning and decide on which tool to use based on user's query. </li>
 <li> Generate a response based on the context provided within the website url. </li>
</ul>
<h3>⚒️ Friendly User Interface</h3>
<ul>
  <li> Utilizes Streamlit interface to build the web application.</li>
  <li> Users can enter a URL to a website and ask questions about the context of that website.</li>
</ul>
<h2>Technical Workflow</h2>
<h3>⚒️ RAG Architecture</h3>
<ul>
  <li> Upon user query; a web URL, a webpagereader is defined to read and convert HTML2Text.</li>
  <li> Embedded text by the Gemini embedding model are stored in a vector store, ChromaDB.</li>
  <li> Then, indexes are created for retrieval and response synthesizer, ready for tool calling.</li>
  <li> LLMSingleSelector decide which route (summary or QnA) upon user query, then generate the response accordingly.</li>
</ul>
<h2>Conclusion</h2>
Throughout this project, I managed to demonstrate the deployment of an easy and smooth bot-like application within a Streamlit page while also showcasing the integration of Google Gemini's llm model, and its language processing capabilities. 
