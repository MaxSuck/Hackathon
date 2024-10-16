from moviepy.editor import VideoFileClip, AudioFileClip
import os, random, io

def generate_video(videoname, audioname):
    try:
        video_path = os.path.join(videoname)
        audio_path = os.path.join(audioname)

        # Load the video and audio files
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Check if the audio file has audio
        if audio is None or audio.duration == 0:
            print("The audio file is empty or not correctly loaded.")
            return

        # Get the duration in seconds
        audio_duration = audio.duration
        video_duration = video.duration

        print(f"Audio duration: {audio_duration}")
        print(f"Video duration: {video_duration}")

        if video_duration > audio_duration:
            # Choose a random start time within the available video length for the audio
            start_time = random.uniform(0, video_duration - audio_duration)

            # Crop the video to the length of the audio
            video = video.subclip(start_time, start_time + audio_duration)
            print(f"Video cropped from {start_time} to {start_time + audio_duration}")
            # Align the audio start time to 0
            audio = audio.set_start(0)
        else:
            # If audio is longer than or equal to the video, start at the beginning
            start_time = 0
            print("Audio duration is longer than or equal to the video duration, using the full video.")

        # Set the audio to the video
        final_video = video.set_audio(audio)

        # Check if the audio is correctly set
        if final_video.audio is None:
            print("Audio is not set correctly on the video.")
        else:
            print("Audio has been successfully attached to the video.")

        # Write the new video file with audio
        final_video.write_videofile("../brainrot.mp4", codec='libx264', audio_codec='aac')

    except Exception as e:
        print(f"An error occurred: {e}")

#def read_file_to_array(file_path):
    try:
        # Read the file and store each line as an element in the list (array)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Strip newline characters from each line
        lines = [line.strip() for line in lines]
        
        return lines  # Return the list of lines
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# def transcription(filename,videoname):

    file_path = os.path.join(filename)

    audio_name = "output.mp3"   # Audio file

    # Read the text lines and count syllables
    words = read_file_to_array(file_path)
    syllable_counts = Transcribe.count_syllables_in_file(file_path)

    print(syllable_counts)

    # Load the audio file
    audio = AudioFileClip(audio_name)
    duration = audio.duration


    # Calculate total syllables
    sum_syllables = 0
    for syl in syllable_counts:
        sum_syllables += syl
        #print(f"Current total syllables: {sum_syllables}")
        #print(f"Syllables in current line: {syl}")

    # Calculate the average duration per syllable
    length_per_syllable = duration / sum_syllables if sum_syllables > 0 else 0

    length_passed = 0
    # Print the duration each line should be displayed based on its syllable count
    for i, word in enumerate(words):
        line_duration = length_per_syllable * syllable_counts[i]
        if (line_duration != 0):
            txtclip = TextClip(str(syllable_counts[i]), fontsize=20, color='black')
            txtclip = txtclip.set_pos('center').set_duration(10)
            clipduration = VideoFileClip("footage.mov").subclip(length_passed, length_passed + line_duration)
            video_with_audio = clipduration.set_audio(audio.subclip(length_passed, length_passed + line_duration))
        
        length_passed += line_duration
        #print(f"Line {i + 1} ('{word}') should be displayed for {line_duration:.2f} seconds")


    

        