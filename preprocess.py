import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv")

# Function to build better instruction-response pairs
def generate_instruction_response(row):
    animal = row['AnimalName']
    symptoms = [row[f"symptoms{i}"] for i in range(1, 6) if pd.notna(row[f"symptoms{i}"])]
    symptom_text = ", ".join(symptoms).lower()

    instruction = f"My {animal.lower()} has {symptom_text}."

    # Smarter responses
    if "vomiting" in symptom_text and "diarrhea" in symptom_text:
        response = "Vomiting and diarrhea together can indicate a serious digestive issue or infection. Ensure your pet stays hydrated and consult a veterinarian immediately."
    elif "weight loss" in symptom_text:
        response = "Weight loss can be a symptom of underlying illness. Please have your pet checked by a veterinarian to rule out serious conditions."
    elif "dehydration" in symptom_text:
        response = "Dehydration is dangerous for pets and can lead to complications. Make sure your pet has access to clean water and seek veterinary care urgently."
    elif "stomach" in symptom_text or "gurgling" in symptom_text:
        response = "Stomach gurgling or unusual sounds could indicate gas or digestive distress. If this is frequent or combined with vomiting, see a vet."
    elif "fever" in symptom_text:
        response = "Fever in pets can be a sign of infection or inflammation. It's best to get a temperature check and see a veterinarian."
    elif "not eating" in symptom_text or "appetite loss" in symptom_text:
        response = "Loss of appetite can have many causes, from stress to serious health issues. Please consult a vet if it continues."
    else:
        response = "These symptoms could point to a range of issues. Please monitor your pet closely and visit a vet for proper diagnosis."

    return {"instruction": instruction, "response": response}

# Generate all rows
records = [generate_instruction_response(row) for _, row in df.iterrows()]

# Save to JSON for training
pd.DataFrame(records).to_json("train_data_richer.json", orient="records", lines=True)

print("✅ Done! Saved to train_data_richer.json")
