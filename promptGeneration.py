from openai import OpenAI
import io, csv, os
client = OpenAI(api_key="")

def chat_with_gpt(prompt):
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

def question_maker(user_prompt, num_of_questions):
    # Open the file in write mode, which will overwrite the file for each run
    with open("question.txt", "w") as file:
        for k in range(int(num_of_questions)):  # Generating 2 questions
            prompt = (
                f"Assume the role of a teacher. Assume proper grammar and make a multiple choice question based on the following prompt. "
                f"Do not do any formatting, assume that the recipient is a python console "
                f"and can only read English. Print the correct answer a line below as 'CORRECT ANSWER: (letter) followed by the description of the choice. Also assume the student does not know the prompt. "
                f"Do not print any blank lines. Prompt: {user_prompt}"
            )
            response = chat_with_gpt(prompt)

            cleaned_response = response.strip()  # Remove any leading/trailing newlines or spaces
            file.write(cleaned_response + "\n") 

