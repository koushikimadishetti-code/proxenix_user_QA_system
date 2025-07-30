 #Install Dependencies
!pip install transformers datasets evaluate

#mount the google drive to save the preprocessed data for later use 
from google.colab import drive
drive.mount('/content/drive')


#Import Libraries
from datasets import load_dataset
import pandas as pd

#load the dataset and save it to drive 
# Load SQuAD dataset
dataset = load_dataset("squad")

# Convert to DataFrames
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])

# Save to Google Drive as CSV
train_df.to_csv("/content/drive/MyDrive/squad_train.csv", index=False)
val_df.to_csv("/content/drive/MyDrive/squad_validation.csv", index=False)

print("Datasets saved to Drive successfully!")

#Load Tokenizer & BERT Model
from transformers import BertTokenizerFast, BertForQuestionAnswering

# Load pretrained tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")


# Tokenize Dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully contained in the context, label it with 0 (token ID for [CLS])
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize train and validation sets
tokenized_train = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_val = dataset["validation"].map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)


# Save Tokenized Dataset to Disk or Google Drive
tokenized_train.save_to_disk("/content/drive/MyDrive/tokenized_squad_train")
tokenized_val.save_to_disk("/content/drive/MyDrive/tokenized_squad_val")


#Load It Next Time Without Re-Tokenizing
from datasets import load_from_disk

tokenized_train = load_from_disk("/content/drive/MyDrive/tokenized_squad_train")
tokenized_val = load_from_disk("/content/drive/MyDrive/tokenized_squad_val")



#to check all the trained questions from the loaded dataset
all_train_questions = dataset["train"]["question"]

print(f"Total number of questions: {len(all_train_questions)}\n")

# Print first 100 questions with numbering
for i, question in enumerate(all_train_questions[:100], start=1):
    print(f"{i}. {question}")



#to check all the validation questions from the loaded dataset
val_questions = dataset["validation"]["question"]

print(f"Total validation questions: {len(val_questions)}\n")

for i, question in enumerate(val_questions[:10], start=1):
    print(f"{i}. {question}")

#Set Training Arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert-qa",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100
)

#Train the Model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()


#Save Your Model and Tokenizer
# Define the directory on your Google Drive to save the model
model_dir = "/content/drive/MyDrive/bert_qa_model"

# Save the trained model
trainer.save_model(model_dir)

# Save the tokenizer
tokenizer.save_pretrained(model_dir)

print("✅ Model and tokenizer saved successfully to Google Drive.")



#To Load Later for Inference
from transformers import BertForQuestionAnswering, BertTokenizerFast

model_dir = "/content/drive/MyDrive/bert_qa_model"

model = BertForQuestionAnswering.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)


#Save TrainingArguments for Later Use
# training_args.save("/content/drive/MyDrive/bert_qa_model/training_args.bin")

import json
# Convert the TrainingArguments object to a dictionary
training_args_dict = training_args.to_dict()

# Define the path to save the JSON file
training_args_path = "/content/drive/MyDrive/bert_qa_model/training_args.json"

# Save the dictionary as a JSON file
with open(training_args_path, "w") as f:
    json.dump(training_args_dict, f, indent=4)

print(f"✅ Training arguments saved successfully to {training_args_path}")



#Define QA Prediction Function
import torch

def predict_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)

    # Move inputs to the same device as the model
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    # Move the input_ids back to CPU for token decoding
    input_ids = inputs["input_ids"][0].cpu()
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer


# Test on Multiple Questions
predictions = []

# Testing on first 100 validation samples
for i in range(100):
    question = dataset["validation"][i]["question"]
    context = dataset["validation"][i]["context"]
    true_answer = dataset["validation"][i]["answers"]["text"][0]

    predicted = predict_answer(question, context)

    predictions.append({
        "question": question,
        "predicted_answer": predicted,
        "true_answer": true_answer
    })

# Print first 5 predictions
for i in range(5):
    print(f"\nQ: {predictions[i]['question']}")
    print(f"Predicted: {predictions[i]['predicted_answer']}")
    print(f"True: {predictions[i]['true_answer']}")


#Evaluate Model (EM & F1)
import evaluate

metric = evaluate.load("squad")

references = [{"id": str(i), "answers": dataset["validation"][i]["answers"]} for i in range(100)]
preds = [{"id": str(i), "prediction_text": predictions[i]["predicted_answer"]} for i in range(100)]

results = metric.compute(predictions=preds, references=references)

print("\nEvaluation Metrics:")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")


#Inference Code to Test the Model for inputing the unknown context which is not in the dataset
def test_custom_qa_loop():
    while True:
        context = input("\nEnter the context passage (or type 'quit' to exit): ").strip()
        if context.lower() == 'quit':
            print("Exiting the QA system.")
            break
        if not context:
            print("Context cannot be empty. Please enter a valid passage.")
            continue

        while True:
            question = input("\nAsk a question about the above context (or type 'done' to input new context): ").strip()
            if question.lower() == 'done':
                break
            if not question:
                print("Question cannot be empty. Please enter a valid question.")
                continue

            predicted_answer = predict_answer(question, context)
            print(f"\nQ: {question}")
            print(f"A: {predicted_answer}")

        cont = input("\nDo you want to continue with another context? (yes/no): ").strip().lower()
        if cont not in ['yes', 'y']:
            print("Goodbye!")
            break

# Run the QA loop
test_custom_qa_loop()
