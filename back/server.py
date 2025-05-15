from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
import requests
import os
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)
    # Set the port to 5000 for local development
    
app.config['UPLOAD_FOLDER'] = './uploads'  # Define upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# 1. Basic CORS setup - Allow all origins during testing
CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins temporarily
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"])

# 2. Global CORS handler for additional headers and preflight requests
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins temporarily
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# 3. Explicit OPTIONS route handler for all routes
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins temporarily
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '86400')  # Cache preflight response for 24 hours
    return response

app.route('/test-cors', methods=['GET', 'POST'])
def test_cors():
    method = request.method
    headers = dict(request.headers)
    data = {}
    
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = dict(request.form)
        # Add file handling if needed
    
    return jsonify({
        "message": "CORS test successful",
        "method": method,
        "headers": headers,
        "data": data
    })

@app.route('/ping')
def ping():
    return jsonify({"message": "Server is alive"}), 200


# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
logger.info(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'

def process_pdf(file):
    """Helper function to handle PDF upload and text extraction."""
    if not file or file.filename == '':
        return None, jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return None, jsonify({'error': 'Invalid file format, only PDFs allowed'}), 400
    
    if request.content_length and request.content_length > 5 * 1024 * 1024:
        return None, jsonify({'error': 'File too large, max 5MB'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
        text = extract_text(file_path)
        if not text.strip():
            return None, jsonify({'error': 'No text could be extracted from the PDF'}), 400
        logger.info(f"Extracted PDF text: {text}")
        return text, None, None
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return None, jsonify({'error': f'PDF processing failed: {str(e)}'}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {str(e)}")

def call_gemini_api(prompt):
    """Helper function to call Gemini API."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set")
        return None, jsonify({'error': 'Server configuration error: API key missing'}), 500

    headers = {'Content-Type': 'application/json'}
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'response_mime_type': 'application/json'}
    }
    
    try:
        response = requests.post(
            f'{GEMINI_API_URL}?key={GEMINI_API_KEY}',
            json=payload,
            headers=headers,
            timeout=30  # Increased timeout from 8 to 30 seconds
        )
        
        if response.status_code != 200:
            error_data = response.json().get('error', {})
            logger.error(f"Gemini API error: {error_data}")
            return None, jsonify({'error': error_data.get('message', 'Gemini API request failed')}), 500
        
        data = response.json()
        candidates = data.get('candidates', [])
        if not candidates:
            logger.error("No candidates in Gemini API response")
            return None, jsonify({'error': 'No content returned from Gemini API'}), 500
        
        content = candidates[0].get('content', {}).get('parts', [])
        if not content:
            logger.error("No content parts in Gemini API response")
            return None, jsonify({'error': 'No content returned from Gemini API'}), 500
        
        text = content[0].get('text')
        if not text:
            logger.error("No text in Gemini API response")
            return None, jsonify({'error': 'No content returned from Gemini API'}), 500
        
        logger.info(f"Raw Gemini API text: {text}")
        try:
            result = json.loads(text)
            logger.info(f"Parsed Gemini API result: {result}")
            if not result or not isinstance(result, dict):
                logger.error("Invalid JSON format from Gemini API: not a dictionary")
                return None, jsonify({'error': 'Invalid response format from Gemini API'}), 500
            
            # Check response format based on endpoint
            if 'notes' in result:
                # For generate-notes endpoint
                return result, None, None
            else:
                # For upload endpoint (solutions)
                if not all(
                    isinstance(value, dict) and "question" in value and "solution" in value
                    for value in result.values()
                ):
                    logger.error("Invalid solution format: missing required fields")
                    return None, jsonify({'error': 'Gemini API returned invalid solution format'}), 500
                return result, None, None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None, jsonify({'error': f'Invalid JSON format from Gemini API: {str(e)}'}), 500
    except requests.RequestException as e:
        logger.error(f"Gemini API request failed: {str(e)}")
        return None, jsonify({'error': f'Gemini API request failed: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Endpoint to solve questions from uploaded PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    text, error_response, status_code = process_pdf(file)
    if error_response:
        return error_response, status_code if status_code else 500

    # Add chunk handling for large text
    if len(text) > 8000:
        logger.info(f"Text is large ({len(text)} chars), summarizing before sending to API")
        text = text[:8000] + "... (text truncated for API consumption)"
        
    prompt = f"""
        You are an expert tutor with a strong background in mathematics and problem-solving. The following text contains questions from a test or questionnaire (TQ). Analyze the text, identify the questions, and provide clear, concise, and accurate solutions for each question. Pay special attention to mathematical problems, ensuring that all calculations are correct and explanations are thorough.

        Format the response as a JSON object where each key is a question number or identifier (e.g., "Q1", "Q2") and the value is an object with "question" (the question text) and "solution" (the answer or explanation). Do not wrap the JSON in any additional structures, arrays, or text. If the questions are not clearly numbered, infer the structure and assign identifiers like "Q1", "Q2", etc. If the text is unclear, provide your best interpretation.

        For mathematical problems, include step-by-step solutions where applicable, and ensure the final answer is clearly stated.

        Example input:
        Q1: What is 2 + 2?
        Q2: Solve for x: 2x + 3 = 7.

        Expected output:
        {{
          "Q1": {{ "question": "What is 2 + 2?", "solution": "2 + 2 = 4" }},
          "Q2": {{ "question": "Solve for x: 2x + 3 = 7", "solution": "2x + 3 = 7\\n2x = 4\\nx = 2" }}
        }}

        Text from file:
        {text}

        Return only the JSON object, without any additional text, markdown, or wrapping.
    """

    result, error_response, status_code = call_gemini_api(prompt)
    if error_response:
        return error_response, status_code if status_code else 500
    
    return jsonify({'solutions': result})

@app.route('/generate-notes', methods=['POST'])
def generate_notes():
    """Endpoint to generate extensive notes from uploaded PDF."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    text, error_response, status_code = process_pdf(file)
    if error_response:
        return error_response, status_code if status_code else 500

    prompt = f"""
        You are an expert educator skilled in creating comprehensive study materials. The following text is extracted from a PDF document. Analyze the text and generate extensive, well-structured notes that summarize and explain the key concepts, ideas, and details in the document. If the text includes questions, pick the topics from the questions and generate a note with it instead of answering the question. The notes should be clear, concise, and suitable for university-level study, organized with headings, bullet points, or numbered lists as appropriate. Focus on clarity, educational value, and retaining all critical information. If the text is unclear or ambiguous, make reasonable interpretations and note any assumptions made.

        Text from file:
        {text}

        Return the response as a JSON object with a single field "notes" containing the generated notes as a string.
        Example format:
        {{
          "notes": "Generated notes text with headings, bullet points, etc."
        }}
    """

    result, error_response, status_code = call_gemini_api(prompt)
    if error_response:
        return error_response, status_code if status_code else 500
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)