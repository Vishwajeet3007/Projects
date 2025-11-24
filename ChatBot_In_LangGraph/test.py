# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load .env file
# load_dotenv()

# # Get the key from environment variable
# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)

# # List available models
# models = genai.list_models()
# for model in models:
#     print(model.name)

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Ensure .env values are loaded
print("KEY LOADED:", os.getenv("GOOGLE_API_KEY"))

# Force REST (avoid v1beta gRPC which causes 404)
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
    transport="rest"
)

# Initialize using correct key name: model_name NOT model
model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")

response = model.generate_content("Hello! Are you available now?")
print(response.text)

