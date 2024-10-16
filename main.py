import promptGeneration, textToSpeech, moviemaker


prompt = input("Input a topic you would like to study/focus on: ")
numGenerated = input("Input number of questions you would like generated: ")

promptGeneration.question_maker(prompt, numGenerated)

textToSpeech.convert("question.txt")

#videoTitle = input("Input video file name (e.g. footage.mov, background.mp4, etc): ")
moviemaker.generate_video("footage.mov","output.mp3")

#convert("question.txt")