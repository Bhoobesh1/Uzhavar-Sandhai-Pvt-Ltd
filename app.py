from flask import Flask, request, jsonify, render_template
import PyPDF2
import faiss
import numpy as np
from openai import OpenAI
import os


# ---------------- APP SETUP ----------------
app = Flask(__name__)
client = OpenAI()

chunks = []
index = None

# ----------- CHAT MEMORY -----------
conversation_memory = []
MAX_MEMORY = 6

# ---------------- SMALL TALK ----------------
def handle_small_talk(user_input):
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    closing = ["bye", "thank you", "thanks", "ok thank you", "ok thanks", "that's all"]

    for g in greetings:
        if text == g or text.startswith(g):
            return "Welcome to Uzhavar Sandhai Pvt Ltd 🌾 How can I help you?"

    for c in closing:
        if c in text:
            return "You're welcome 😊 Feel free to ask anytime."

    return None

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("uzhavar.html")

# ---------------- CHUNKING ----------------
def make_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------- EMBEDDINGS ----------------
def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)

    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ---------------- LOAD DEFAULT PDF ----------------
def load_default_pdf(pdf_path):
    global chunks, index, conversation_memory

    conversation_memory = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    if not text.strip():
        print("❌ No readable text in default PDF")
        return

    chunks = make_chunks(text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    print("✅ Default PDF loaded successfully")

# ---------------- ASK QUESTION ----------------
@app.route("/ask", methods=["POST"])
def ask():
    global conversation_memory

    question = request.json.get("question", "")

    if not question:
        return jsonify({"answer": "Please ask a question."})

    # Small talk
    small_talk_response = handle_small_talk(question)
    if small_talk_response:
        return jsonify({"answer": small_talk_response})

    if index is None:
        return jsonify({"answer": "Document not loaded."})

    # Embed question
    q_embed = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    q_embed = np.array([q_embed]).astype("float32")

    distances, indices = index.search(q_embed, 3)

    context = "\n\n".join([chunks[i] for i in indices[0]])

    # -------- MEMORY PROMPT --------
    memory_text = ""
    for m in conversation_memory:
        memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n\n"

    prompt = f"""
You are a helpful assistant.

Previous conversation:
{memory_text}

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    # Save memory
    conversation_memory.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_memory) > MAX_MEMORY:
        conversation_memory.pop(0)

    return jsonify({"answer": answer})

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    load_default_pdf("data/document.pdf")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
