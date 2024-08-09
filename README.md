<h1 align="center"> LlamaIndex RAG WebPageReader QnA Bot with Gemini 🤖 </h1>

<h5 align="center">Keywords: Python | LlamaIndex | RAG | LLM | Gemini-Pro | NLP | API | Streamlit</h5>
<h2>Project Description</h2>
Developed the simplest form of agentic RAG (Retrieval-Augmented-Generation) with a router. 
Agentic RAG is a framework designed to build research agents capable of tool use, reasoning and decision-making with given data.
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
  <li> The conversation interface utilizes Streamlit chat-message interface.</li>
</ul>
<h2>Technical Workflow</h2>
<h3>⚒️ Google's Gemini Secret Key</h3>
<ul>
  <li> Generate the API key that was used to call on the llm model.</li>
  <li> Storing the secret key in an environment file for a more efficient program architecture.</li>
</ul>
<h2>Conclusion</h2>
Throughout this project, I managed to demonstrate the deployment of an easy and smooth bot-like application within a Streamlit page while also showcasing the integration of Google Gemini's llm model, and its language processing capabilities. 
