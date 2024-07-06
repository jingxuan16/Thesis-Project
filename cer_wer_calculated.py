import re
import whisper
from jiwer import cer, wer

# Initialize the Whisper model
model = whisper.load_model("medium.en")

# Define the reference sentences
reference_sentences = {
    '1': "The service to Tokyo departs in 10min",
    '2': "The express to New York City is now boarding from gate nine",
    '3': "The train to Amsterdam is delayed by one hour"
}

# Initialize dictionaries for CER and WER
cer_dict = {'1': [], '2': [], '3': []}
wer_dict = {'1': [], '2': [], '3': []}

# Function to clean sentences
def clean_sentence(sentence):
    return re.sub(r'[^\w\s]', '', sentence).lower().strip()

# Function to process a group of files
def process_files(group_num, start, end):
    reference = reference_sentences[group_num]
    reference_cleaned = clean_sentence(reference)

    for i in range(start, end + 1):
        # Construct the file name
        file_name = f"{group_num}-{i}.wav"

        # Transcribe the audio file with specified language
        result = model.transcribe(file_name, language="en")
        transcription = result["text"]

        # Clean the transcription
        transcription_cleaned = clean_sentence(transcription)

        # Calculate CER and WER
        cer_error = cer(reference_cleaned, transcription_cleaned)
        wer_error = wer(reference_cleaned, transcription_cleaned)

        # Add to error rate lists
        cer_dict[group_num].append((i, cer_error))
        wer_dict[group_num].append((i, wer_error))

        # Print results
        print(f"Transcription for {file_name}: {transcription}")
        print(f"CER ({file_name}): {cer_error}")
        print(f"WER ({file_name}): {wer_error}")
        print()

# Process each group of files
process_files('1', 1, 5)
process_files('2', 1, 5)
process_files('3', 1, 5)

# Print the entire error rate lists
print(f"CER List: {cer_dict}")
print(f"WER List: {wer_dict}")
