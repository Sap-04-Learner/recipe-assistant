# 👨‍🍳 AI Chef Assistant (Agentic RAG Recipe App)

An intelligent, fully local AI culinary assistant that suggests recipes based on your available ingredients and guides you step-by-step through the cooking process. 

Unlike traditional chatbots, this project uses an **Agentic RAG (Retrieval-Augmented Generation)** architecture. A background "Silent Router" intelligently decides when to search the vector database for recipes and when to simply converse, ensuring a natural and dynamic cooking experience.

## ✨ Features

- **🧠 Agentic Router:** Uses LLM intent analysis to decide whether to search the recipe database or just chat (bypasses searches for navigational words like "yes", "next", or "done").
- **📖 Step-by-Step Guidance:** Never overwhelms you with a wall of text. The Chef walks you through recipes one step at a time, waiting for your confirmation before proceeding.
- **🥑 Searchable Ingredient Drawer:** Automatically extracts thousands of unique ingredients from the dataset and provides a searchable UI drawer to quickly add items to your pantry.
- **💬 Modern Streaming UI:** ChatGPT-style interface with real-time text streaming (via Server-Sent Events) and Markdown rendering.
- **🔒 100% Local & Private:** Runs entirely locally using [Ollama](https://ollama.com/), `llama3`, and ChromaDB. No API keys or internet required.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Vanilla), `marked.js`
- **Backend:** Python, Flask
- **Vector Database:** ChromaDB
- **LLM Engine:** Ollama (`llama3` for chat & routing, `nomic-embed-text` for vector embeddings)
- **Data Processing:** Pandas, Jupyter Notebook
- **Dataset:** [Food.com Recipes and User Interactions (Kaggle)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
1. **Python 3.8+**
2. **Ollama:** [Download and install Ollama](https://ollama.com/)
3. **Pull Required Local Models:**
   Open your terminal and run:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

## 🚀 Installation & Setup
1. **Clone the repository and navigate to the directory**
     ```bash
       git clone https://github.com/Sap-04-Learner/recipe-assistant.git
       cd recipe-assistant
     ```
2. **Install Python Dependencies**
    ```bash
      python -m pip install -r ../requirements.txt
    ```
3. **Prepare the Dataset**
    - Download the raw dataset from Kaggle.
    - Create a `data/` folder in your project directory.
    - Extract `RAW_recipes.csv` into the `data/` folder.
4. **Clean Data & Build the Vector Database**
    - Open data-exploration.ipynb in Jupyter and run all cells to clean the dataset and output optimized_recipes_for_rag.csv.
    - Run the indexer to embed the recipes into ChromaDB:
        ```bash
          python indexer.py
        ```
        *(Note: This step may take some time depending on your hardware, as it generates vector embeddings for the recipes).*
5. **Start the Application**
    ```bash
      python app.py
    ```
6. **Open in Browser**
    - Go to http://localhost:5000 to start cooking!

## 📂 Project Structure
```
├── app.py                     # Main Flask web server & streaming SSE logic
├── config.py                  # Configuration paths and LLM settings
├── generator.py               # (Terminal version) Chat loop & Agent logic
├── indexer.py                 # Embeds dataset into ChromaDB
├── retriever.py               # Semantic search logic for finding recipes
├── data-exploration.ipynb     # Jupyter Notebook for data cleaning
├── templates/
│   └── index.html             # The frontend chat UI
├── data/                      # Folder containing CSV datasets
└── recipe_db/                 # Auto-generated ChromaDB vector storage
```

## 🧠 How the Agentic RAG Works
1. **User Input:** The user types ingredients or replies to a step.
2. **Silent Router:** `app.py` intercepts the message. A background LLM prompt determines if the user is listing ingredients (triggering a database search) or just chatting/navigating.
3. **Context Injection:** If a search is triggered, retriever.py queries ChromaDB. The top 3 recipes are injected temporarily into the LLM's context window.
4. **Generation:** `llama3` evaluates the retrieved recipes against the user's pantry. If it's a match, it suggests them. If the user selects a recipe, it enters "Step-by-Step" mode using a strict system prompt.




