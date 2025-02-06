from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import requests
from prompts import INTERVIEW_PROMPTS
from dotenv import load_dotenv
import uuid
from transcribe import transcribe_audio_with_speakers
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.urandom(24)

API_KEY = os.getenv('BLAND_AI_API_KEY')
API_URL = 'https://api.bland.ai/v1/calls'

@app.route('/')
def home():
    return render_template('home.html', interview_prompts=INTERVIEW_PROMPTS)

@app.route('/start-call', methods=['POST'])
def start_call():
    phone_number = request.form.get('phone_number')
    interview_type = request.form.get('interview_type')
    voice = request.form.get('voice')

    if not phone_number:
        flash('Phone number is required!', 'error')
        return redirect(url_for('home'))

    headers = {
        'Authorization': f'{API_KEY}'
    }

    # Generate a unique call ID
    call_id = str(uuid.uuid4())
    print(call_id)
    # Bland AI API Call Docs: https://docs.bland.ai/api-v1/post/calls
    data = {
        "call_id": call_id,
        "phone_number": phone_number,
        "from": None,
        "task": INTERVIEW_PROMPTS.get(interview_type, INTERVIEW_PROMPTS["default"]),
        "model": "enhanced",
        "language": "en",
        "voice": voice,  # Use the selected voice
        "voice_settings": {},
        "pathway_id": None,
        "local_dialing": False,
        "max_duration": "30",
        "answered_by_enabled": True,
        "wait_for_greeting": True,
        "record": True,
        "amd": False,
        "interruption_threshold": 200,  # Technically this should be dynamic based on the context. 
        "voicemail_message": None,
        "temperature": None,
        "transfer_phone_number": None,
        "transfer_list": {},
        "metadata": {},
        "pronunciation_guide": [],
        "start_time": None,
        "request_data": {},  # set of variables available to the voice agent. 
        "tools": [],
        "dynamic_data": [],
        "analysis_preset": None,
        "analysis_schema": {},
        "webhook": None,
        "calendly": {},
        "background_track": "cafe",
        "record": True,  # Record the call for later processing into content. 
        # "keywords": ["{company_name}"]
    }

    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        flash('Call started successfully!', 'success')
        # Store call_id in session for later use if needed
        session['last_call_id'] = call_id
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if response is not None:
            print(f"Response content: {response.text}")
        flash(f'An error occurred: {e}', 'error')

    return redirect(url_for('home'))

@app.route('/get-recording/<call_id>', methods=['GET'])
def get_recording(call_id):
    headers = {
        'Authorization': f'{API_KEY}'
    }
    
    url = f"https://api.bland.ai/v1/calls/{call_id}/recording"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    
    transcript = transcribe_audio_with_speakers(response.json()["url"])
    print(transcript)

    return jsonify({'transcript': transcript})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)  # Comment this out for production
    # pass  # Gunicorn will handle running the app
