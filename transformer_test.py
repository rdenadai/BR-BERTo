from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./BR_BERTo",
    tokenizer="./BR_BERTo", topk=5
)

fill_mask("eu gosto muito de <mask>")