# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a huggingface course my whole life")

print(res)

#test