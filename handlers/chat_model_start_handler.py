from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from pyboxen import boxen

def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))
    


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages,**kwargs) :
        print(messages)