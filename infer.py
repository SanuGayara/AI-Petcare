from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("petcare-model")
model = T5ForConditionalGeneration.from_pretrained("petcare-model")

def ask_pet_assistant(query):
    prompt = "Symptom: " + query
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

while True:
    msg = input("You: ")
    if msg.lower() in {"exit", "quit"}:
        break
    print("PetCareAI:", ask_pet_assistant(msg))
