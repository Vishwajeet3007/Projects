from google.generativeai import GenerativeModel

import google.generativeai as genai

genai.configure(api_key="AIzaSyDH1MvDXs5Ooj5i6mqOsWFi8T1aE7GBg00")

models = genai.list_models()
for model in models:
    print(model.name)