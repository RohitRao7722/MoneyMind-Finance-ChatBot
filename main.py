import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import FAISS
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime

# Constants for paths
VECTORSTORE_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.json"
CHAT_HISTORY_PATH = "chat_history.json"

# Streamlit configuration
st.set_page_config(page_title="Interact with **:blue[MoneyMind]** to unlock financial insights and guidance", page_icon=":books:")


# Load VectorStore
def load_vectorstore():
    vectorstore = FAISS.load_local(
        VECTORSTORE_INDEX_PATH, 
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
        allow_dangerous_deserialization=True
    )
    with open(METADATA_PATH, "r") as file:
        texts = json.load(file)
    vectorstore.texts = texts
    return vectorstore

# Load and save chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_chat_history():
    chat_history = load_chat_history()
    if "current_chat" in st.session_state and st.session_state.current_chat:
        chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.current_chat
        })
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(chat_history, f)

def update_chat_history():
    chat_history = load_chat_history()
    if "selected_chat_index" in st.session_state and st.session_state.current_chat:
        selected_chat_index = st.session_state.selected_chat_index
        if 0 <= selected_chat_index < len(chat_history):
            chat_history[selected_chat_index]["messages"] = st.session_state.current_chat
            with open(CHAT_HISTORY_PATH, "w") as f:
                json.dump(chat_history, f)

# Function to get AI response
def get_response(context, question, model):
    # Initialize chat session if not already active
    if "chat_session" not in st.session_state or st.session_state.chat_session is None:
        st.session_state.chat_session = model.start_chat(history=[])

    # Initialize current chat if not already set
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    # Collect previous conversation to maintain context
    chat_history_context = ""
    for past_query, past_response in st.session_state.current_chat:
        chat_history_context += f"User asked: {past_query}\nAI responded: {past_response}\n\n"

    # Compose prompt with structured response guidance and accumulated history context
    prompt_template = f"""
    You are a knowledgeable assistant who responds to user queries with detailed and context-aware answers. When users ask for elaboration, provide relevant, informative responses based on previous interactions without asking for more context. Use the following structure:

    - Provide an *in-depth explanation* or *details*.
    - Relate the answer to previous interactions if applicable.
    - Break down key points in bullet points if necessary.

    *Previous Conversations*:
    {chat_history_context}

    *Current Context*: {context}
    *User's Question*: {question}

    Respond with relevant details and examples based on the user's question.
    """

    try:
        # Generate response
        response = st.session_state.chat_session.send_message(prompt_template)

        # Append new question and response to current chat history
        st.session_state.current_chat.append((question, response.text))
        return response.text

    except AttributeError:
        st.warning("Chat session could not be initialized. Please restart the app.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Sidebar UI for chat history and settings
# Sidebar UI for chat history and settings
def sidebar_ui():
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
    st.session_state.api_key = api_key
    
    st.sidebar.markdown("### Previous Chats")
    chat_history = load_chat_history()

    # Display each previous chat with the first question as its title
    chat_options = ["Start a New Chat"] + [
        f"{i+1}. {chat['messages'][0][0]}" if chat["messages"] else f"{i+1}. (No question)"
        for i, chat in enumerate(chat_history)
    ]
    
    selected_chat = st.sidebar.selectbox("Select a previous chat", chat_options, index=0)
    if selected_chat == "Start a New Chat":
        st.session_state.current_chat = []  # Clear current chat history
        st.session_state.selected_chat = None
        st.session_state.chat_session = None
    else:
        selected_chat_index = int(selected_chat.split(".")[0]) - 1
        st.session_state.selected_chat = chat_history[selected_chat_index]["messages"]
        st.session_state.selected_chat_index = selected_chat_index
        st.session_state.current_chat = st.session_state.selected_chat.copy()


# Main working process
def working_process(generation_config):
    # Ensure the API key is set
    if not st.session_state.api_key:
        st.warning("Please enter a valid API key in the sidebar.")
        return
    
    # Configure the GenAI model
    try:
        genai.configure(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return

    # Initialize the model with instructions
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    vectorstore = st.session_state['vectorstore']

    # Display each interaction in its own chat block
    if "current_chat" in st.session_state:
        for user_query, ai_response in st.session_state.current_chat:
            with st.chat_message("Human"):
                st.markdown(user_query)
            if ai_response:
                with st.chat_message("AI"):
                    st.markdown(ai_response)

    # Capture new user query
    user_query = st.chat_input("Enter Your Query....")
    if user_query:
        # Display the user's query in its own block immediately
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Append the new query to current chat history without a response yet
        st.session_state.current_chat.append((user_query, None))

        try:
            # Retrieve relevant context from the vectorstore and generate response
            relevant_content = vectorstore.similarity_search(user_query, k=10)
            result = get_response(relevant_content, user_query, model)

            # Update the current chat with the AI response
            st.session_state.current_chat[-1] = (user_query, result)  # Update with the AI's response
            with st.chat_message("AI"):
                st.markdown(result)

        except Exception as e:
            st.warning(f"Error generating response: {e}")

    # Save or update chat history based on whether it's a new or existing chat session
    if st.session_state.new_chat:
        save_chat_history()
        st.session_state.new_chat = False  # Reset to ensure new chat only triggers once
    else:
        update_chat_history()

# Main Function
def main():
    load_dotenv()
    st.header("Interact with **:blue[MoneyMind]** to unlock financial insights and guidance")

    # Initialize session state attributes if not present
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading preprocessed data..."):
            st.session_state.vectorstore = load_vectorstore()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    # Ensure new_chat is initialized
    if "new_chat" not in st.session_state:
        st.session_state.new_chat = True  # Or set to False if this isn't a new chat

    sidebar_ui()
    if st.session_state.vectorstore:
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8000,
        }
        working_process(generation_config)

if __name__ == "__main__":
    main()