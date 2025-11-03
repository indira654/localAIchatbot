from langchain_ollama import OllamaLLM
model = OllamaLLM(model="tinyllama")
result = model.invoke(input="Hello, I'm Indu!")
print(result)