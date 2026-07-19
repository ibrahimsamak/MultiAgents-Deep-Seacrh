import tempfile

from dotenv import load_dotenv
from openai import OpenAI

from Knowledge.knowledge_base import build_news_vector_db
from agents.LLMAgent import LLMAgent
from OpenAILLM import OpenAILLM
from agents.RagAgent import RagAgent
from Supervisor import Supervisor
from agents.SearchAgent import SearchAgent
from agents.ProductAgent import ProductAgent
from tools.search_tools import DuckDuckGoNewsTool
from tools.product_tools import DuckDuckGoProductSearch


class VoiceAssistant:
    """Holds the supervisor and provides the speech helpers used by the web UI:
    Whisper transcription for voice input and OpenAI TTS for spoken answers."""

    def __init__(self, supervisor, client=None, voice="alloy"):
        self.supervisor = supervisor
        self.client = client or OpenAI()
        self.voice = voice

    def transcribe_file(self, path):
        """Transcribe an audio file with Whisper. The web UI records audio in the
        browser and hands us the resulting file."""
        with open(path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        return transcript.text.strip()

    def _tts_wav(self, text):
        """Call OpenAI TTS and return the synthesized speech as WAV bytes."""
        response = self.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        return response.read()

    def synthesize_to_file(self, text):
        """Synthesize `text` to a temporary .wav file and return its path, for
        playback in the browser."""
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(path, "wb") as out:
            out.write(self._tts_wav(text))
        return path


def build_default_assistant():
    """Wire up the assistant: OpenAI LLM + Chroma-backed news RAG, orchestrated
    by an LLM Supervisor that plans and picks the next agent each step."""
    load_dotenv()
    client = OpenAI()

    llm = OpenAILLM(client=client)
    llm_agent = LLMAgent(llm)
    rag_agent = RagAgent(build_news_vector_db(), llm=llm)
    search_agent = SearchAgent(DuckDuckGoNewsTool(), llm=llm)
    product_agent = ProductAgent(DuckDuckGoProductSearch())

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
            "name": "product_shopper",
            "description": (
                "Searches the web for a product, compares prices across listings, "
                "picks the cheapest, and converts that price to Canadian dollars "
                "(CAD). Use for shopping and 'cheapest'/'best price'/'how much"
                "'/'price in CAD' questions about a product to buy."
            ),
            "agent": product_agent,
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
