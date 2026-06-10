from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

def load_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    print("API Key Loaded:", api_key is not None)

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )