"""
Simple Python application entry point.
This is a basic Flask web application.
"""

from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Welcome to TNT Back!',
        'status': 'running'
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
