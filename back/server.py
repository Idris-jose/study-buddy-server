from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
import requests
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS to allow React frontend to communicate
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gemini API configuration
GEMINI_API_KEY = os.getenv('VITE_GEMINI_API_KEY')  # Use environment variable
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'

def process_pdf(file):
    """Helper function to handle PDF upload and text extraction."""
    if not file or file.filename == '':
        return None, jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return None, jsonify({'error': 'Invalid file format, only PDFs allowed'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Extract text from PDF
        text = extract_text(file_path)
        if not text.strip():
            return None, jsonify({'error': 'No text could be extracted from the PDF'}), 400
        return text, None
    except Exception as e:
        return None, jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def call_gemini_api(prompt):
    """Helper function to call Gemini API."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'response_mime_type': 'application/json'}
    }
    response = requests.post(
        f'{GEMINI_API_URL}?key={GEMINI_API_KEY}',
        json=payload,
        headers=headers
    )
    
    if response.status_code != 200:
        error_data = response.json().get('error', {})
        return None, jsonify({'error': error_data.get('message', 'Gemini API request failed')}), 500
    
    data = response.json()
    content = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    if not content:
        return None, jsonify({'error': 'No content returned from Gemini API'}), 500
    
    try:
        result = json.loads(content)
        if not result or not isinstance(result, dict):
            return None, jsonify({'error': 'Invalid response format from Gemini API'}), 500
        return result, None
    except json.JSONDecodeError as e:
        return None, jsonify({'error': f'Invalid JSON format from Gemini API: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Endpoint to solve questions from uploaded PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    text, error_response = process_pdf(file)
    if error_response:
        return error_response

    # Prepare prompt for solving questions
    prompt = f"""
        You are an expert tutor with a strong background in mathematics and problem-solving. The following text contains questions from a test or questionnaire (TQ). Analyze the text, identify the questions, and provide clear, concise, and accurate solutions for each question. Pay special attention to mathematical problems, ensuring that all calculations are correct and explanations are thorough. Format the response as a JSON object where each key is a question number or identifier (e.g., "Q1", "Q2") and the value is an object with "question" (the question text) and "solution" (the answer or explanation). If the questions are not clearly numbered, infer the structure and assign identifiers. If the text is unclear, provide your best interpretation.

        For mathematical problems, include step-by-step solutions where applicable, and ensure that the final answer is clearly stated.

        For best results, expect the text to be structured like:
        Q1: What is 2 + 2?
        Q2: Solve for x: 2x + 3 = 7.
        Or similar clear formats.

        Text from file:
        {text}

        Return the response in the following format:
        {{
          "Q1": {{ "question": "Question text", "solution": "Solution text" }},
          "Q2": {{ "question": "Question text", "solution": "Solution text" }},
          ...
        }}
    """

    result, error_response = call_gemini_api(prompt)
    if error_response:
        return error_response
    
    return jsonify({'solutions': result})

@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """Endpoint to generate extensive notes from uploaded PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    text, error_response = process_pdf(file)
    if error_response:
        return error_response

    # Prepare prompt for generating notes
    prompt = f""".
        You are an expert educator skilled in creating comprehensive study materials. The following text is extracted from a PDF document. Analyze the text and generate extensive, well-structured notes that summarize and explain the key concepts, ideas, and details in the document.if the text include questions pick the topics from the questions and generate a note with it instaed of answering the question.  The notes should be clear, concise, and suitable for university-level study, organized with headings, bullet points, or numbered lists as appropriate. Focus on clarity, educational value, and retaining all critical information. If the text is unclear or ambiguous, make reasonable interpretations and note any assumptions made.

        Text from file:
        {text}

        Return the response as a JSON object with a single field "notes" containing the generated notes as a string.
        Example format:
        {{
          "notes": "Generated notes text with headings, bullet points, etc."
        }}
    """

    result, error_response = call_gemini_api(prompt)
    if error_response:
        return error_response
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)