import pickle
import numpy as np

model = pickle.load(open("chat_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
responses = pickle.load(open("responses.pkl", "rb"))

def get_response(user_input):
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)[0]
    return np.random.choice(responses[prediction])

print("HR Assistant: Hello! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("HR Assistant: Goodbye!")
        break
    response = get_response(user_input)
    print("HR Assistant:", response)
