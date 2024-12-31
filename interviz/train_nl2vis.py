import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer,T5TokenizerFast, AdamW
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Define a Custom Dataset for NL to Visualization
class NLToVisualizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        question = item["query"]
        target_code = json.dumps(item["code"])

        # Tokenize and encode the question and target code
        inputs = self.tokenizer.encode_plus(
            "translate English to visualization code: " + question,
            padding="max_length",
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        )
        target_inputs = self.tokenizer.encode_plus(
            target_code,
            padding="max_length",
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        target_attention_mask = target_inputs["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask
        }
            
        # # Tokenize and encode the question and target code
        # inputs = self.tokenizer.encode_plus(
        #     "translate English to visualization code: " + question,
        #     target_code,
        #     padding="max_length",
        #     max_length=2048,
        #     truncation=True,
        #     return_tensors="pt"
        # )
        
        # input_ids = inputs["input_ids"].squeeze()
        # attention_mask = inputs["attention_mask"].squeeze()
        
        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "target_ids": input_ids,  # For conditional generation, target is the same as input
        #     "target_attention_mask": attention_mask
        # }

# Load and Preprocess the Dataset

with open("try_train.json", "r") as f:
    train_dataset = json.load(f)
with open("try_dev.json", "r") as f:
    dev_dataset = json.load(f)

train_data=NLToVisualizationDataset(train_dataset)
dev_data=NLToVisualizationDataset(dev_dataset)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)  # batch_size=8
dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)



# Initialize the T5 Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Prepare the Training Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Train the Model
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 1
for epoch in range(num_epochs):
    print(f"This is epoch {epoch+1}.")
    model.train()
    train_loss = 0.0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attention_mask,
            use_cache=False
        )
        print("Outputs:", outputs)

        # Compute the loss
        # loss = outputs.loss
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        print("Loss:", loss)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_dataloader)

    # Evaluate on the validation set
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_attention_mask,
                use_cache=False
            )
            
            # Compute the loss
            loss = outputs.loss
            valid_loss += loss.item()

        # Calculate average validation loss for the epoch
        avg_valid_loss = valid_loss / len(dev_dataloader)

    # Print the average training and validation loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Valid Loss: {avg_valid_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("model/trained_nl2vis_model")

       
