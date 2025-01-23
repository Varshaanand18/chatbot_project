from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# Function to load the model and tokenizer
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# Function to generate the response from the model
def generate_response(input_text, model_path):
    tokenizer, model = load_model(model_path)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function to compare two models' outputs
def chatbot():
    user_input = input("You: ")

    # Use the fine-tuned model
    fine_tuned_model_path = "./fine_tuned_model"  # Update the path if necessary

    # Generate response using the fine-tuned model
    response = generate_response(user_input, fine_tuned_model_path)

    print(f"Response from fine-tuned model: {response}")

# Fine-Tuning Section

def fine_tune_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load a dataset (e.g., QA dataset for fine-tuning)
    dataset = load_dataset('csv', data_files='C:/Users/varsanan/OneDrive - Capgemini/Desktop/qa_data.csv', split='train')

    # Preprocess the dataset: Tokenize inputs and outputs
    def preprocess_function(examples):
        inputs = examples['question']
        targets = examples['answer']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Split the dataset into train and evaluation datasets
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    print("Fine-tuning complete!")

# Run the chatbot
if __name__ == "__main__":
    # Comment out the fine-tuning line after running it once
    #fine_tune_model()  # Uncomment this line to fine-tune the model initially
    chatbot()
