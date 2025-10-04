# tnt-back

A simple Python backend application using Flask.

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JonFjz/tnt-back.git
cd tnt-back
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Local Development
```bash
python main.py
```

The application will be available at `http://localhost:5000`

#### Using Docker
```bash
# Build the Docker image
docker build -t tnt-back .

# Run the container
docker run -p 5000:5000 tnt-back
```

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint

## Project Structure

```
tnt-back/
├── main.py           # Main application entry point
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker configuration
├── .gitignore        # Git ignore rules
└── README.md         # This file
```