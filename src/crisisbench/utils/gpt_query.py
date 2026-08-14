from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def query_gpt(conversation_history, system_prompt, model="gpt-4"):
    openai_client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_history}
    ]
    response = openai_client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=messages
    )
    return response.choices[0].message.content.strip()
