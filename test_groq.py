import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env")
    exit()

print("API key found.")
print("Sending test request to Groq...")

try:
    client = Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=1
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ],
        temperature=0
    )

    print("\n✅ Groq is working!")
    print("Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\n❌ Groq error:")
    print(type(e).__name__)
    print(e)