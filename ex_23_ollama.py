import ollama

#response = ollama.chat(model='qwen2:0.5b', messages=[{'role': 'user', 'content': 'Schreibe Java Code der den Stratey Pattern erklärt'}])
response = ollama.chat(model='qwen2:0.5b', messages=[{'role': 'user', 'content': 'Write a class which explains the strategy pattern in java'}])

print(response['message']['content'])

response = ollama.chat(model='qwen2:0.5b', messages=[{'role': 'user', 'content': 'In which way is a stack used when implementing the strategy pattern in Java?'}])

print(response['message']['content'])