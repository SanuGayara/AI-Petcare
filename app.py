import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model
tokenizer = T5Tokenizer.from_pretrained("petcare-model")
model = T5ForConditionalGeneration.from_pretrained("petcare-model")

# Chat function
def get_advice(symptom_input):
    prompt = "Symptom: " + symptom_input
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Gradio UI
iface = gr.Interface(
    fn=get_advice,
    inputs=gr.Textbox(placeholder="Describe your pet's symptoms..."),
    outputs="text",
    title="🐾 PetCare AI Assistant",
    description="Enter your pet's symptoms and get friendly advice. For safety, always follow up with a real veterinarian."
)

iface.launch()
