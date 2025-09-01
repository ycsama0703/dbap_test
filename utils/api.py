from openai import OpenAI

import os


def get_response(prompt):
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")  # Use yoru own API Key

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    return response.choices[0].message.content

def get_openai_response(prompt):

    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") # Use yoru own API Key
    )

    response = client.responses.create(
    model="gpt-4o-mini",
    input="write a haiku about ai",
    store=True,
    )

    return response.output_text




if __name__ == "__main__":
    prompt = input()
    print(get_response(prompt))

    