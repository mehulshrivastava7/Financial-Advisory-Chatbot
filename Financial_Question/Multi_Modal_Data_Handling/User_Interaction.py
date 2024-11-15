"""
This code is by Nitin.
This script processes user input in various formats (text, PDF, audio) to extract financial information using language detection, translation, and natural language processing.
This script adheres to PEP 8 style
"""

import os
import json
import pytesseract
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
from pdf2image import convert_from_path
from transformers import MarianMTModel, MarianTokenizer
from PyPDF2 import PdfFileReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Uncomment the following line to set OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")


def detect_language(text):
    """
    Detect the language of the provided text.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: Detected language code or None if detection fails.
    """
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print(f"Language detection error: {e}")
        return None


def translate_to_english(text, src_lang):
    """
    Translate text from the source language to English.

    Args:
        text (str): The text to translate.
        src_lang (str): The source language code.

    Returns:
        str: Translated text in English or None if translation fails.
    """
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]
    except Exception as e:
        print(f"Translation to English error: {e}")
        return None


def translate_from_english(text, tgt_lang):
    """
    Translate text from English to the target language.

    Args:
        text (str): The text to translate.
        tgt_lang (str): The target language code.

    Returns:
        str: Translated text in the target language or None if translation fails.
    """
    try:
        model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]
    except Exception as e:
        print(f"Translation from English error: {e}")
        return None


def extract_financial_info(text):
    """
    Extract financial information from the provided text using OpenAI's API.

    Args:
        text (str): The text from which to extract financial information.

    Returns:
        dict: Extracted financial information in JSON format or None if extraction fails.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None

    prompt = f"""
    Extract the following details from the provided text about stock transactions:
    
    1. **Asset Name** (e.g., company or stock name)
    2. **Number of Shares**
    3. **Purchase Date**
    4. **Purchase Price**
    
    Text: '{text}'
    
    If any detail is missing, leave that field blank. Respond with the extracted details in this JSON format:
    {{
      "Asset Name": "...",
      "Number of Shares": "...",
      "Purchase Date": "...",
      "Purchase Price": "..."
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can switch to "gpt-3.5-turbo" if needed
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0  # Set to 0 for more deterministic output
        )
        extracted_data = response.choices[0].message['content'].strip()

        # Parse the JSON string to a Python dictionary
        try:
            extracted_json = json.loads(extracted_data)
            return extracted_json
        except json.JSONDecodeError:
            print("Failed to parse the response as JSON.")
            print("Response received:", extracted_data)
            return None
    except Exception as e:
        ```python
        print(f"Error in LLM extraction: {e}")
        return None


def check_missing_data(financial_info):
    """
    Check for any missing financial data in the extracted information.

    Args:
        financial_info (dict): The extracted financial information.

    Returns:
        list: A list of keys for which the values are missing.
    """
    missing_data = [key for key, value in financial_info.items() if not value]
    return missing_data


def generate_back_prompt(missing_data):
    """
    Generate a prompt asking the user for the missing financial data.

    Args:
        missing_data (list): A list of missing data keys.

    Returns:
        str: A prompt string requesting the missing information.
    """
    prompt = "Please provide the following missing information: "
    prompt += ", ".join(missing_data)
    return prompt


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file. If text extraction fails, use OCR to extract text from images.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: Extracted text or None if extraction fails.
    """
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PdfFileReader(file)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extractText()

        # If text extraction fails, use OCR
        if not text.strip():
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img)

        return text
    except Exception as e:
        print(f"PDF text extraction error: {e}")
        return None


def speech_to_text(audio_file_path, lang_code):
    """
    Convert speech from an audio file to text.

    Args:
        audio_file_path (str): The path to the audio file.
        lang_code (str): The language code for speech recognition.

    Returns:
        str: Recognized text or None if conversion fails.
    """
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=lang_code)
        return text
    except Exception as e:
        print(f"Speech-to-text error: {e}")
        return None


def text_to_speech(text, lang_code, output_file):
    """
    Convert text to speech and save it to an audio file.

    Args:
        text (str): The text to convert to speech.
        lang_code (str): The language code for the speech.
        output_file (str): The path to save the output audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(output_file)
    except Exception as e:
        print(f"Text-to-speech error: {e}")


def process_user_input(user_input, input_type="text", user_lang=None):
    """
    Process user input based on its type (text, PDF, audio) and extract financial information.

    Args:
        user_input (str): The input provided by the user.
        input_type (str): The type of input ('text', 'pdf', 'audio').
        user_lang (str): The language code of the user input (if known).
    """
    if input_type == "text":
        text = user_input
    elif input_type == "pdf":
        text = extract_text_from_pdf(user_input)
        if not text:
            print("Failed to extract text from PDF.")
            return
    elif input_type == "audio":
        text = speech_to_text(user_input, user_lang)
        if not text:
            print("Failed to convert speech to text.")
            return
    else:
        print("Unsupported input type.")
        return

    # Detect language if not provided
    if not user_lang:
        user_lang = detect_language(text)
        if not user_lang:
            print("Failed to detect language.")
            return

    print(f"Detected Language: {user_lang}")

    # Translate to English if necessary
    if user_lang != "en":
        text_in_english = translate_to_english(text, user_lang)
        if not text_in_english:
            print("Failed to translate to English.")
            return
    else:
        text_in_english = text

    print(f"Text in English: {text_in_english}")

    # Extract financial information
    financial_info = extract_financial_info(text_in_english)
    if not financial_info:
        print("Failed to extract financial information.")
        return

    print(f"Extracted Financial Information: {financial_info}")

    # Check for missing data
    missing_data = check_missing_data(financial_info)

    if missing_data:
        # Generate back-prompt
        back_prompt_en = generate_back_prompt(missing_data)
        print(f"Back Prompt in English: {back_prompt_en}")

        # Translate back prompt to user's language if necessary
        if user_lang != "en":
            back_prompt_user_lang = translate_from_english(back_prompt_en, user_lang)
            if not back_prompt_user_lang:
                print("Failed to translate back prompt.")
                return
        else:
            back_prompt_user_lang = back_prompt_en

        print(f"Back Prompt in User's Language: {back_prompt_user_lang}")

        # Convert back-prompt to speech if the input was audio
        if input_type == "audio":
            output_audio_file = "back_prompt.mp3"
            text_to_speech(back_prompt_user_lang, user_lang, output_audio_file)
            print(f"Back prompt audio saved to {output_audio_file}")
            # Here you can play the audio file if necessary
        else:
            # Get user input for missing data
            user_response = input(back_prompt_user_lang)
            # Recursively process the new input
            process_user_input(user_response, input_type="text", user_lang=user_lang)
    else:
        # Format the financial information into a table or desired format
        print("Final Financial Information:")
        for key, value in financial_info.items():
            print(f"{key}: {value}")

        # Proceed with further processing (e.g., portfolio optimization)
        # ...

# Example usage
if __name__ == "__main__":
    # Example text input
    user_input_text = "मैंने 1 जनवरी, 2020 को एसबीआई के 50 शेयर खरीदे हैं।"
    process_user_input(user_input_text)

    # Example PDF input
    # user_input_pdf = "financial_statement.pdf"
    # process_user_input(user_input_pdf, input_type="pdf")

    # Example audio input
    # user_input_audio = "financial_info.wav"
    # process_user_input(user_input_audio, input_type="audio", user_lang="hi-IN")
