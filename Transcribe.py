import re

def count_syllables(word):
    # Regular expression pattern to identify syllables, with case-insensitive flag applied directly
    word = word.lower()
    syllable_pattern = r'[aeiouy]{1}[aeiouy]*[^aeiouy]|[aeiouy]{1}(?![aeiouy])'
    
    # Find all matches and count them
    syllables = re.findall(syllable_pattern, word, re.IGNORECASE)
    count = len(syllables)
    
    # Adjust for certain cases (e.g., silent 'e')
    if word.endswith('e') and not word.endswith(('le', 'ye')):
        count -= 1
    if count < 1:  # Ensure there's at least one syllable
        count = 1
        
    return count

def syllables_per_line(text):
    # Split the text into lines and count syllables for each line
    lines = text.splitlines()
    return [sum(count_syllables(word) for word in re.findall(r'\b\w+\b', line)) for line in lines]

def count_syllables_in_file(file_path):
    """Counts syllables in each line of a file and returns a list of counts."""
    try:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            text = file.read()
        
        syllable_counts = syllables_per_line(text)
        #print("hello")
        
        return syllable_counts  # Return the list of syllable counts
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Example usage
# file_path = "question.txt"  # Replace with your text file path
# syllable_counts = count_syllables_in_file(file_path)
# print(syllable_counts)


# Example usage
# file_path = "question.txt"  # Replace with your text file path
# lines_array = read_file_to_array(file_path)
# print(lines_array)



