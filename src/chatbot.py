import os
import uuid
from typing import List, Dict
from datetime import datetime

from ollama import Client
from pydantic import BaseModel, Field

# Load environment variables from venv.env
def load_env(path: str) -> None:
  if not os.path.isfile(path):
    return
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, val = line.split("=", 1)
      os.environ.setdefault(key.strip(), val.strip().strip("'\""))

load_env(os.path.join(os.path.dirname(__file__), "..", "venv.env"))

class Message(BaseModel):
  id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  role: str
  content: str
  timestamp: datetime = Field(default_factory=datetime.utcnow)

  def to_ollama(self) -> Dict:
    return {"role": self.role, "content": self.content}

class ChatResponse(BaseModel):
  message: Message
  branch_id: str = ""
  branch_name: str = "main"
  total_messages: int = 0

class Ollama_ChatBot:
  DEFAULT_MODEL = "gpt-oss:120b"
  DEFAULT_HOST = "https://ollama.com"

  def __init__(self, model: str | None = None, host: str | None = None, system_prompt: str | None = None, stream: bool = True) -> None:
    self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
    self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
    self.stream = stream
    self._messages: list[Message] = []

    if system_prompt:
      self._system_prompt = system_prompt
      self._messages.append(Message(role="system", content=system_prompt))

    headers = {}
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if api_key:
      headers["Authorization"] = f"Bearer {api_key}"

    if self.model == "deepseek-r1:1.5b":
      self._client = Client()
    else:
      self._client = Client(host=self.host, headers=headers)

  def chat(self, user_input: str) -> ChatResponse:
    self._messages.append(Message(role="user", content=user_input))
    assistant_content = self._call_llm([m.to_ollama() for m in self._messages])
    assistant_msg = Message(role="assistant", content=assistant_content)
    self._messages.append(assistant_msg)
    return ChatResponse(message=assistant_msg, branch_id="main", branch_name="main", total_messages=len(self._messages))

  def _call_llm(self, messages: List[Dict]) -> str:
    if self.stream:
      return self._stream_response(messages)
    response = self._client.chat(self.model, messages=messages, stream=False)
    return response["message"]["content"]

  def _stream_response(self, messages: List[Dict]) -> str:
    full_text = ""
    for part in self._client.chat(self.model, messages=messages, stream=True):
      token = part["message"]["content"]
      print(token, end="", flush=True)
      full_text += token
    print()
    return full_text
