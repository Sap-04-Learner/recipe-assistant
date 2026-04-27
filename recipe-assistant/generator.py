import ollama
import json
import re
from retriever import get_recipe_recommendations

chat_history =[]

# --- BRAIN 1: THE CHEF PERSONA ---
system_instruction = """You are a conversational, expert culinary assistant.
CRITICAL RULES:
1. BRAINSTORMING: If the user lists 1 or 2 ingredients, do NOT suggest recipes yet. Act like a chef. Say "Great, we have fish. Do you have any citrus, rice, or herbs to go with it?" Wait for them to finalize their list.
2. QUALITY CONTROL: When the system provides [DATABASE RECIPES], evaluate them. If they match the user's ingredients, suggest 2-3 of them naturally.
3. BE HONEST: If they do NOT match well, say: "I checked my recipe book, but I don't have a perfect match. However, we could try..."
4. STEP-BY-STEP MODE: If the user asks to cook a specific recipe, DO NOT list all steps at once. 
   - Say: "Let's start! Step 1 is [Step 1]. Let me know when you are ready for the next step."
   - Wait for the user to say "next", "done", or "ready".
   - Then provide Step 2, and so on. Never overwhelm them with the whole recipe.
"""
chat_history.append({'role': 'system', 'content': system_instruction})

# --- BRAIN 2: THE SILENT ROUTER ---
def check_if_ready_to_search(user_input):
    """A silent, fast background check to see if we should hit the database."""
    router_prompt = """You are an intent analyzer.
    If the user has provided ingredients and explicitly or implicitly wants a recipe right now, output EXACTLY this JSON: {"search": true, "query": "ingredient1, ingredient2"}
    If the user is just answering a question, brainstorming, or saying hello, output EXACTLY: {"search": false, "query": ""}
    """
    
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': router_prompt},
                {'role': 'user', 'content': user_input}
            ],
            format='json' # Forces the LLM to output pure JSON data
        )
        
        raw_output = response['message']['content']
        # Clean up markdown formatting if the model accidentally includes it
        raw_output = re.sub(r'```json\n?|```\n?', '', raw_output).strip()
        
        return json.loads(raw_output)
    except Exception as e:
        # If it fails, default to not searching to be safe
        return {"search": False, "query": ""}

# --- THE MAIN CHAT LOOP ---
def chat_loop():
    print("\n=======================================================")
    print("👨‍🍳 The Chef is in the kitchen. (Type 'exit' to quit)")
    print("=======================================================\n")
    print("Chef: Hello! I'm your kitchen assistant. What ingredients are we working with today?\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("\nChef: Happy cooking! Goodbye!")
                break
            
            # 1. Ask the silent Router if we should search the database
            intent = check_if_ready_to_search(user_input)
            
            # 2. Save user message to history
            chat_history.append({'role': 'user', 'content': user_input})
            temp_messages = chat_history.copy()
            
            # 3. Execute search ONLY if the Router says yes
            if intent.get('search'):
                search_query = intent.get('query', user_input)
                print(f"\n[System: 🔍 Chef is quietly checking the pantry for '{search_query}'...]")
                
                recipes = get_recipe_recommendations(search_query, top_k=3)
                
                if recipes:
                    # Local context variable, NOT global
                    current_context = "\n[DATABASE RECIPES - Evaluate these to see if they match the user's ingredients:]\n"
                    for r in recipes:
                        current_context += f"- Name: {r.get('name')} | Ingredients: {r.get('ingredients')} | Steps: {r.get('steps')}\n"
                    
                    # Inject ONLY into the temporary message sent to the LLM for this turn
                    temp_messages[-1]['content'] = f"{user_input}\n\n{current_context}"

            print("\nChef: ", end="", flush=True)
            
            # 4. Stream the final response
            response_stream = ollama.chat(
                model='llama3', 
                messages=temp_messages, 
                stream=True
            )
            
            full_response = ""
            for chunk in response_stream:
                word = chunk['message']['content']
                print(word, end="", flush=True)
                full_response += word
            
            print("\n\n" + "-"*55 + "\n")
            
            # 5. Save the assistant's response to actual history
            chat_history.append({'role': 'assistant', 'content': full_response})

        except KeyboardInterrupt:
            print("\n\nExiting kitchen...")
            break
        except Exception as e:
            print(f"\n[ERROR] Something went wrong: {e}")

if __name__ == "__main__":
    chat_loop()