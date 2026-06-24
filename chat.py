import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def chat(model, messages):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def main():
    print("CLI Chat App - OpenRouter")
    print("Type 'quit' to exit, 'switch' to change model\n")
    
    models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "meta-llama/llama-3.2-3b-instruct:free"
    ]
    
    print("Available models:")
    for i, m in enumerate(models):
        print(f"{i+1}. {m}")
    
    choice = input("\nChoose model (1-3): ").strip()
    model = models[int(choice)-1]
    print(f"\nUsing: {model}\n")
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "switch":
            for i, m in enumerate(models):
                print(f"{i+1}. {m}")
            choice = input("Choose model (1-3): ").strip()
            model = models[int(choice)-1]
            print(f"Switched to: {model}\n")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("Assistant: ", end="", flush=True)
        response = chat(model, messages)
        print(response)
        
        messages.append({"role": "assistant", "content": response})
        print()

if __name__ == "__main__":
    main()
