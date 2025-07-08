import streamlit as st
import json
import requests

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# Streamlit UI
st.title("💬 Chat with Ollama (LLaMA 3)")

user_input = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if user_input:
        payload = {
            "model": "llama3.2",
            "prompt": user_input,
            "stream": False
        }

        with st.spinner("Generating response..."):
            try:
                response = requests.post(OLLAMA_URL, json=payload)

                if response.status_code == 200:
                    text = response.text.strip()

                    # Try parsing as a single JSON first
                    try:
                        result_json = json.loads(text)
                        result = result_json.get("response", "No response received.")
                    except json.JSONDecodeError:
                        # If multiple JSON objects or newline-delimited JSON
                        lines = text.splitlines()
                        responses = [json.loads(line) for line in lines if line.strip()]
                        result = "\n".join(r.get("response", "") for r in responses)

                    st.success(result)

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
