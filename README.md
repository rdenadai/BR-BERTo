# BR-BERTo
Transformer model for Portuguese language (Brazil pt_BR)

The first model trained (which is a RoBERTa model), can be found on tags page: https://github.com/rdenadai/BR-BERTo/releases/tag/0.1

The full and latest model, should be downloaded from **Huggingface** page: https://huggingface.co/rdenadai/BR_BERTo

### Params (latest model)

Trained on a corpus of 6_993_330 sentences.

- Vocab size: 150_000
- RobertaForMaskedLM  size : 512
- Num train epochs: 3
- Time to train: ~10days (on GCP with a Nvidia T4)

I follow the great tutorial from HuggingFace team:

[How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)
