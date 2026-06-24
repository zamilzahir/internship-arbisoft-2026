# Prompt Examples

## 1. Zero-Shot
No examples given, just a direct question.

**Prompt:**
"Classify this sentence as positive or negative: I love this product!"

**Response:**
Positive

---

## 2. Few-Shot
Give examples before asking.

**Prompt:**
"Classify these sentences:
- I love this product! → Positive
- This is terrible → Negative
- Amazing experience! → Positive

Now classify: The service was okay."

**Response:**
Neutral

---

## 3. System Prompt
Set a role or persona first.

**Prompt:**
System: You are a helpful assistant that explains things simply to beginners.
User: What is an API?

**Response:**
An API is like a
cat > prompts.md << 'EOF'
# Prompt Examples

## 1. Zero-Shot
No examples given, just a direct question.

**Prompt:**
"Classify this sentence as positive or negative: I love this product!"

**Response:**
Positive

---

## 2. Few-Shot
Give examples before asking.

**Prompt:**
"Classify these sentences:
- I love this product! → Positive
- This is terrible → Negative
- Amazing experience! → Positive

Now classify: The service was okay."

**Response:**
Neutral

---

## 3. System Prompt
Set a role or persona first.

**Prompt:**
System: You are a helpful assistant that explains things simply to beginners.
User: What is an API?

**Response:**
An API is like a waiter in a restaurant. You tell the waiter what you want, they go to the kitchen and come back with your order.

---

## 4. Structured Output
Ask for a specific format like JSON.

**Prompt:**
"Return a JSON object with name, age, and city for a fictional person."

**Response:**
{"name": "John Smith", "age": 28, "city": "New York"}
