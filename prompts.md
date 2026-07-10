# Prompts Log

## Entry 1
**Prompt:** "Help me set up my project to call the OpenRouter API — install the right packages, create a `.env` file to store the API key safely, an `.env.example` for GitHub, and make sure `.env` is gitignored."

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Scaffold the project environment for OpenRouter API testing

**Outcome:**
Step 1 — Go to your project
cd ~/Desktop/internship-arbisoft-2026
Step 2 — Install required packages
pip3 install requests python-dotenv
Step 3 — Create your .env file (to store API key safely)
echo "OPENROUTER_API_KEY=<REDACTED_API_KEY>" > .env
Step 4 — Create .env.example (safe to push to GitHub)
echo "OPENROUTER_API_KEY=your_api_key_here" > .env.example
Step 5 — Make sure .env is ignored by git
echo ".env" >> .gitignore
Paste what you see after each step and I'll give you the full CLI app code next!

> Note: the original key you pasted was redacted here since a live API key shouldn't be repeated in a committed file.

---

## Entry 2
**Prompt:** "I'm getting an error when testing the first model — what does it mean?"

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Debug a rate-limit error on the first free model tested

**Outcome:**
The error is 429 — rate limited. The free model is temporarily busy. It says `retry_after_seconds: 27`.

Let's try model 2 instead:
```python
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY')
r = requests.post('https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'model': 'meta-llama/llama-3.2-3b-instruct:free', 'messages': [{'role': 'user', 'content': 'What is an API?'}]})
print(r.status_code)
print(r.json())
"
```
Paste what it prints!

---

## Entry 3
**Prompt:** "Both models are rate limited now, what's going on?"

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Understand why both free models were rate limited on a new account

**Outcome:**
Both are rate limited (429). This is a new account with no credits. You need to add a small amount of credits to OpenRouter to use it.

Go to openrouter.ai → Credits in the left sidebar and add $5 — that's more than enough for all your deliverables.

Or alternatively, try the Google Gemma free model which might not be rate limited:
```python
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY')
r = requests.post('https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'model': 'google/gemma-3-4b-it:free', 'messages': [{'role': 'user', 'content': 'What is an API?'}]})
print(r.status_code)
print(r.json())
"
```
Which would you prefer — add credits or try another free model?

---

## Entry 4
**Prompt:** "to test three models I need to pay?"

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Clarify whether payment was required just to test free models

**Outcome:**
No! The free models just have rate limits — you hit them too fast. Wait 1-2 minutes and try again. Run:
```python
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY')
r = requests.post('https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'model': 'meta-llama/llama-3.2-3b-instruct:free', 'messages': [{'role': 'user', 'content': 'What is an API?'}]})
print(r.status_code)
print(r.json())
"
```
Wait a minute first then run it. It should work — you just sent too many requests in a short time!

---

## Entry 5
**Prompt:** "Still getting 429 after waiting — what else can I try?"

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Find a workaround for persistent rate limiting on a brand-new OpenRouter account

**Outcome:**
Still 429. The issue is you're a new account with no credits added yet. OpenRouter requires at least a small amount of credits even for free models on new accounts.

You have two options:

**Option A — Add $5 credits (quickest)**
Go to openrouter.ai → Credits and add $5. Free models won't actually charge you, but the account needs credits to unlock rate limits.

**Option B — Use a different free provider**
Try Groq instead — it's completely free with no credit requirement:
1. Go to console.groq.com
2. Sign up with Google
3. Get API key
4. It has llama3 and mistral for free with generous limits

---

## Entry 6
**Prompt:** "I switched to Groq and got an API key. Now build me a simple CLI app in Python that lets me chat with the Groq API from the terminal."

**Tool:** Claude
**Model:** Claude Sonnet
**Purpose:** Build a working command-line chat client against the Groq API to replace the rate-limited OpenRouter setup

**Outcome:**
Here's a simple CLI chat client. Add `GROQ_API_KEY=your_key_here` to your `.env` file first, then run `python3 chat.py`.

```python
# chat.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def main():
    print("Groq CLI Chat — type 'exit' to quit\n")
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
        )

        reply = response.choices[0].message.content
        print(f"Groq: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
```

Install the SDK first: `pip3 install groq python-dotenv`. This keeps conversation history in memory so follow-up questions have context, and exits cleanly on `exit` or `quit`.