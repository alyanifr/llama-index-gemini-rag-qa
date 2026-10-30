import sys
import types
 
_MODULE = "langchain_community.chat_models.vertexai"
 
try:
    __import__(_MODULE)
except ModuleNotFoundError:
    class ChatVertexAI:  # placeholder; never instantiated by this project
        pass
 
    shim = types.ModuleType(_MODULE)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE] = shim