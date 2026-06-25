import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def main():
    print("CLI Chat App - Groq")
    print("Type 'quit' to exit, 'switch' to change model\n")
    
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b"
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
        response = chat(model, messages)
        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
