from transformers import pipeline

generator = pipeline("text2text_generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

res = generator( "Generate a multiple choice quiz about discrete mathematics",  num_return_sequences=1 )
print(res)

