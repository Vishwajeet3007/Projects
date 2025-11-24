# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# print("Listing available models...")
# models = genai.list_models()

# for m in models:
#     print("⚡", m.name)

# model = genai.GenerativeModel("gemini-1.5-flash-002")
# resp = model.generate_content("Hello! Are you now working properly?")
# print("\nResponse =>", resp.text)

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.0-flash")
response = model.generate_content("Hello! Are you working properly now?")
print("\nResponse =>", response.text)
