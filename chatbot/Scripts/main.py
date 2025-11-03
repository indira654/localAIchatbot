try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError as e:
    import sys
    raise SystemExit(
        f"Missing package: {e}. Activate your venv and install packages:\n"
        "  source chatbot/Scripts/activate   # Git Bash\n"
        "  python -m pip install langchain-core langchain-ollama\n\n"
        f"Running Python: {sys.executable}"
    )

template = """
Answer the question below
Here is the conversation History: {context}
Question: {question}
Answer:
"""

model = OllamaLLM(model="tinyllama")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break

        try:
            result = chain.invoke({"context": context, "question": user_input})
        except Exception as e:
            print("Error invoking model:", e)
            print("Try a smaller model, free GPU memory, or run on CPU.")
            continue

        # result can be a rich object depending on the LLM wrapper; convert to str if needed
        bot_text = result if isinstance(result, str) else str(result)
        print("Bot:", bot_text)

        # update conversation history
        context += f"\nUser: {user_input}\nBot: {bot_text}"

def main():
    handle_conversation()

if __name__ == "__main__":
    main()