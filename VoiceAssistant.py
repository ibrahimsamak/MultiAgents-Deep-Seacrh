import tempfile

from dotenv import load_dotenv
from openai import OpenAI

from Knowledge.knowledge_base import build_news_vector_db
from agents.LLMAgent import LLMAgent
from OpenAILLM import OpenAILLM
from agents.RagAgent import RagAgent
from Supervisor import Supervisor
from agents.SearchAgent import SearchAgent
from tools.search_tools import DuckDuckGoNewsTool


class VoiceAssistant:
    def __init__(self, supervisor, client=None, voice="alloy"):
        self.supervisor = supervisor
        self.client = client or OpenAI()
        self.voice = voice

    def transcribe_file(self, path):
        with open(path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        return transcript.text.strip()

    def _tts_wav(self, text):
        response = self.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        return response.read()

    def synthesize_to_file(self, text):
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(path, "wb") as out:
            out.write(self._tts_wav(text))
        return path


def build_default_assistant():
    load_dotenv()
    client = OpenAI()

    llm = OpenAILLM(client=client)
    llm_agent = LLMAgent(llm)
    rag_agent = RagAgent(build_news_vector_db(), llm=llm)
    search_agent = SearchAgent(DuckDuckGoNewsTool(), llm=llm)

    agents = [
        {
            "name": "web_news",
            "description": (
                "Searches the LIVE internet (DuckDuckGo News) for the most "
                "recent, breaking, or up-to-the-minute news on any topic. Use "
                "for 'latest'/'today'/'current' events and anything time-sensitive."
            ),
            "agent": search_agent,
        },
        {
            "name": "news_rag",
            "description": (
                "Answers from a small local sample news database (markets, tech, "
                "sports, health, climate, entertainment). Curated but may be "
                "stale; prefer web_news for genuinely recent events."
            ),
            "agent": rag_agent,
        },
        {
            "name": "llm",
            "description": (
                "General-purpose assistant for reasoning, explanations, "
                "definitions, and questions not tied to fresh news."
            ),
            "agent": llm_agent,
        },
    ]

    supervisor = Supervisor(llm, agents)
    return VoiceAssistant(supervisor, client=client)
