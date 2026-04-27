from flask import Flask, render_template, request, jsonify, Response
import ollama
import json
import re
import pandas as pd

from retriever import get_recipe_recommendations
from config import OLLAMA_MODEL, CSV_PATH

app = Flask(__name__)

# --- IN-MEMORY CHAT HISTORY ---
chat_history =[]

system_instruction = """You are a conversational, expert culinary assistant.
CRITICAL RULES:
1. BRAINSTORMING: If the user lists 1 or 2 ingredients, do NOT suggest recipes yet. Say "Great, we have fish. Do you have any citrus, rice, or herbs to go with it?" Wait for them to finalize.
2. QUALITY CONTROL: Evaluate [DATABASE RECIPES]. If they match well, suggest 1-2.
3. BE HONEST: If they do NOT match well, say: "I checked my recipe book, but I don't have a perfect match."
4. STEP-BY-STEP MODE: If the user asks to cook a specific recipe, DO NOT list all steps at once. 
   - Say: "Let's start! Step 1 is[Step 1]. Let me know when you are ready for the next step."
   - Wait for the user to say "next", "done", or "ready".
   - Then provide Step 2, and so on. Never overwhelm them.
"""

chat_history.append({'role': 'system', 'content': system_instruction})

# --- EXTRACT *ALL* UNIQUE INGREDIENTS ON STARTUP ---
print("Extracting ALL unique ingredients from dataset...")
try:
    df = pd.read_csv(CSV_PATH)
    all_ingredients_set = set()
    for item in df['clean_ingredients'].dropna():
        # Split by comma, strip spaces, and convert to lowercase
        ingredients =[i.strip().lower() for i in str(item).split(',')]
        all_ingredients_set.update([i for i in ingredients if i]) # Add to set if not empty
    
    # Sort them alphabetically
    ALL_INGREDIENTS = sorted(list(all_ingredients_set))
    print(f"Successfully loaded {len(ALL_INGREDIENTS)} unique ingredients!")
except Exception as e:
    print(f"Could not load ingredients: {e}")
    ALL_INGREDIENTS =["chicken", "salt", "butter", "onion", "garlic", "milk", "water", "cheese"]

# --- ROUTER LOGIC ---
def check_if_ready_to_search(user_input):
    # 1. HARD BYPASS: Do not search if user is just navigating or using short words
    nav_words =['yes', 'no', 'ok', 'done', 'next', 'ready', 'yep', 'yeah', 'proceed']
    cleaned = re.sub(r'[^a-zA-Z\s]', '', user_input).strip().lower()
    
    if cleaned in nav_words or len(cleaned) <= 3:
        return {"search": False, "query": ""}

    # 2. Ask the LLM Router
    router_prompt = """You are an intent analyzer.
    ONLY if the user provides NEW ingredients and explicitly wants a recipe, output EXACTLY this JSON: {"search": true, "query": "ingredient1, ingredient2"}
    If the user is answering a preference question (e.g., "straightforward is good", "baked"), choosing an option you gave them, or saying they are done, output EXACTLY: {"search": false, "query": ""}
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': router_prompt},
                {'role': 'user', 'content': user_input}
            ],
            format='json'
        )
        raw_output = response['message']['content']
        raw_output = re.sub(r'```json\n?|```\n?', '', raw_output).strip()
        return json.loads(raw_output)
    except:
        return {"search": False, "query": ""}

# --- FLASK ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/ingredients", methods=["GET"])
def get_ingredients():
    return jsonify({"ingredients": ALL_INGREDIENTS})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    chat_history.append({'role': 'user', 'content': user_input})
    temp_messages = chat_history.copy()
    
    intent = check_if_ready_to_search(user_input)
    
    system_notification = ""
    if intent.get('search'):
        search_query = intent.get('query', user_input)
        system_notification = f"🔍 Chef quietly checked the pantry for: {search_query}"
        recipes = get_recipe_recommendations(search_query, top_k=3)
        if recipes:
            current_context = "\n[DATABASE RECIPES - Evaluate these to see if they match the user's ingredients:]\n"
            for r in recipes:
                current_context += f"- Name: {r.get('name')} | Ingredients: {r.get('ingredients')} | Steps: {r.get('steps')}\n"
            temp_messages[-1]['content'] = f"{user_input}\n\n{current_context}"

    # STREAMING FUNCTION
    def generate():
        # Yield system message first if it exists
        if system_notification:
            yield f"data: {json.dumps({'type': 'system', 'content': system_notification})}\n\n"
            
        # Yield AI text chunk by chunk
        response = ollama.chat(model=OLLAMA_MODEL, messages=temp_messages, stream=True)
        ai_reply = ""
        for chunk in response:
            word = chunk['message']['content']
            ai_reply += word
            yield f"data: {json.dumps({'type': 'chunk', 'content': word})}\n\n"
            
        chat_history.append({'role': 'assistant', 'content': ai_reply})
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)