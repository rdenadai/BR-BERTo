from pathlib import Path
import warnings
import pandas as pd
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch


warnings.filterwarnings("ignore")


class EsperantoDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, length=10000):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.len = length
        self.size = 0
        self.dataset = pd.read_csv(self.file_path, header=None, sep="\\n", iterator=True)

    def __len__(self):
        return self.len

    def preprocess(self, text):
        batch_encoding = self.tokenizer(str(text).strip(), add_special_tokens=True, truncation=True, max_length=512)
        return torch.tensor(batch_encoding["input_ids"])

    def __getitem__(self, i):
        if self.size == self.len:
            self.dataset = pd.read_csv(self.file_path, header=None, sep="\\n", iterator=True)
            self.size = 0
        self.size += 1

        phrase = self.dataset.get_chunk(1).to_numpy()[0][0]
        example = self.preprocess(phrase)
        return example


# Check that PyTorch sees it
print("CUDA:", torch.cuda.is_available())
corpus_length = 6_993_330 # fazer um wc -l para ver a qtde de linhas
vocab_size = 150_000

# Dataset files
# --------------------------------------------------
paths = [str(x) for x in Path("./").glob("**/corpus.txt")]

# Byte Level Tokernize
# --------------------------------------------------
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
# Save files to disk
tokenizer.save_model("BR_BERTo")
# Test
tokenizer = ByteLevelBPETokenizer(
    "./BR_BERTo/vocab.json",
    "./BR_BERTo/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
print(tokenizer.encode("gostei muito dessa ideia".lower()).tokens)

# Model type
# --------------------------------------------------
config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=8,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)
print("Params: ", model.num_parameters())
tokenizer = RobertaTokenizerFast.from_pretrained("./BR_BERTo", max_len=512)

# Dataset load
# --------------------------------------------------
dataset = EsperantoDataset(
    tokenizer=tokenizer,
    file_path="./corpus.txt",
    length=corpus_length
)

# Start training
# --------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./BR_BERTo",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=6,
    do_train=True,
    save_steps=500_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model("./BR_BERTo")
