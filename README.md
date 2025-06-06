# RAG Chatbot (VPBank Hackathon)

## Overview
This project is a Retrieval Augmented Generation (RAG) chatbot developed for the VPBank Hackathon. It is designed to scrape information from websites, process and understand the textual content, generate vector embeddings for efficient retrieval, and store this knowledge in a vector database. Users can then interact with the chatbot, which leverages this information to provide relevant and context-aware answers.

## Features
- **Web Scraping**: Dynamically crawls websites to gather data using Beautiful Soup.
- **Advanced Text Processing**: Employs semantic chunking (potentially using Langchain) and keyword extraction for better understanding of content.
- **Embeddings Generation**: Utilizes Gemini AI to create high-quality vector embeddings from text chunks.
- **Vector Storage & Search**: Leverages PostgreSQL with the PgVector extension for storing and performing efficient similarity searches on embeddings.
- **RAG Chatbot**: Provides an interactive chat interface where the bot answers queries based on the retrieved information, powered by Gemini AI.
- **RESTful API**: Exposes a comprehensive set of API endpoints built with FastAPI for scraping, searching, and chatting.
- **Monitoring & Logging**: Features detailed logging for all operations, health check endpoints, and a statistics API for data tracking.
- **Optimized Performance**: Incorporates optimizations such as batch processing for embeddings, database connection pooling, caching mechanisms, background tasks for scraping, and retry logic for API calls.

## Architecture & Tech Stack
The project is built upon a modern and robust technology stack:

-   **Web Framework**: FastAPI
-   **Database**: PostgreSQL with PgVector extension (for vector storage and similarity search)
-   **AI Models**: Gemini AI (for embeddings generation and chat response generation)
-   **Web Scraping**: Beautiful Soup
-   **Text Processing**: Langchain (conceptualized for semantic chunking and text processing workflows)
-   **Containerization**: Docker and Docker Compose (for easy setup and deployment)

### Key Components:
1.  **Web Scraping Service**: Responsible for crawling websites, extracting content, and handling HTML parsing.
2.  **Text Processing Service**: Performs semantic chunking of text, potentially using TF-IDF similarity or other advanced methods, and extracts keywords.
3.  **Embeddings Service**: Generates vector embeddings from text chunks using the Gemini embedding model, with batch processing for efficiency.
4.  **Vector Store Service**: Manages the storage of documents and chunks in PostgreSQL and facilitates vector similarity searches using PgVector.
5.  **RAG Chatbot Service**: Orchestrates the chat flow by retrieving relevant context from the vector store, constructing prompts, and interacting with Gemini AI to generate responses. Manages conversation history.

## Project Structure
```
rag_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main FastAPI application
│   ├── config.py              # Environment configurations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py        # Database connection setup
│   │   └── schemas.py         # Pydantic models for data validation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── web_scraper.py     # Web crawling logic (Beautiful Soup)
│   │   ├── text_processor.py  # Text processing and chunking
│   │   ├── embeddings.py      # Embeddings generation (Gemini)
│   │   ├── vector_store.py    # Interaction with PgVector
│   │   └── chatbot.py         # Core chatbot logic
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── scraping.py    # API endpoints for data scraping
│   │   │   ├── search.py      # API endpoints for search
│   │   │   └── chat.py        # API endpoints for chat
│   │   └── dependencies.py    # Dependency injection for API routes
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # Logging configuration
│       └── helpers.py         # Utility functions
├── migrations/
│   └── init_pgvector.sql     # SQL script for PgVector initialization
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_embeddings.py
│   └── test_chatbot.py
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables file
├── docker-compose.yml          # Docker Compose configuration for services (e.g., Postgres)
├── Dockerfile                  # Dockerfile for containerizing the application
└── README.md                   # This file
```

## Prerequisites
Before you begin, ensure you have the following installed:
-   Python (3.8+ recommended)
-   Docker and Docker Compose
-   Git
-   A Gemini API Key

## Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url> # Replace <your-repo-url> with the actual repository URL
    cd rag_chatbot
    ```

2.  **Create and Activate a Virtual Environment:**
    -   For Linux/macOS:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   For Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Open the `.env` file and update it with your specific configurations, including your `GEMINI_API_KEY` and any database credentials if they differ from defaults.

## Running the Application

1.  **Start the PostgreSQL Database:**
    Use Docker Compose to start the PostgreSQL service, which should include the PgVector extension.
    ```bash
    docker-compose up postgres -d
    ```
    *The `migrations/init_pgvector.sql` script is intended to set up the PgVector extension. Ensure your `docker-compose.yml` is configured to run this or apply it manually if needed.*

2.  **Run the FastAPI Application:**
    ```bash
    python -m app.main
    ```
    The application will typically be accessible at `http://localhost:8000` (or the port configured in your `.env` file). The API documentation (Swagger UI) should be available at `http://localhost:8000/docs`.

## API Endpoints and Usage

The application exposes several RESTful API endpoints for interaction:

### Scraping Endpoints
-   **`POST /api/v1/scraping/scrape-website`**: Initiates scraping of a given website. This is a background task.
    **Request Body:**
    ```json
    {
      "url": "https://example.com",
      "max_depth": 2,
      "max_pages": 10
    }
    ```
    **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/scraping/scrape-website" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com", "max_depth": 2, "max_pages": 10}'
    ```

-   **`GET /api/v1/scraping/documents`**: Retrieves a list of scraped documents.
-   **`DELETE /api/v1/scraping/documents/{id}`**: Deletes a specific scraped document by its ID.

### Search Endpoints
-   **`POST /api/v1/search/semantic`**: Performs a semantic search based on a query.
    **Request Body:**
    ```json
    {
      "query": "information about product X",
      "max_results": 5,
      "similarity_threshold": 0.7
    }
    ```
    **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/search/semantic" \
    -H "Content-Type: application/json" \
    -d '{"query": "information about product X", "max_results": 5, "similarity_threshold": 0.7}'
    ```

-   **`GET /api/v1/search/stats`**: Provides statistics about the vector store.

### Chat Endpoints
-   **`POST /api/v1/chat/message`**: Sends a message to the chatbot and receives a response.
    **Request Body:**
    ```json
    {
      "message": "Tell me about product ABC",
      "conversation_id": null  // Can be null for a new conversation or an existing ID
    }
    ```
    **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/chat/message" \
    -H "Content-Type: application/json" \
    -d '{"message": "Tell me about product ABC", "conversation_id": null}'
    ```

-   **`GET /api/v1/chat/conversation/{id}`**: Retrieves a specific conversation by ID.
-   **`DELETE /api/v1/chat/conversation/{id}`**: Deletes a specific conversation by ID.

## Workflow Overview

The RAG chatbot operates through a sequence of interconnected phases:

1.  **Scraping Phase**:
    -   A user initiates scraping via the API, providing a target URL.
    -   The `WebScraper` service crawls the website up to the specified depth and page limits.
    -   Raw HTML content is processed, and relevant text is extracted.
    -   The `TextProcessor` service segments the extracted text into meaningful chunks (semantic chunking).
    -   The `Embeddings` service converts these text chunks into numerical vector representations (embeddings) using Gemini AI.
    -   Finally, the `VectorStore` service persists these chunks and their corresponding embeddings into the PostgreSQL database (PgVector).

2.  **Search Phase (Context Retrieval)**:
    -   When a user asks a question or provides a search query, the system first generates an embedding for this input query using Gemini AI.
    -   This query embedding is then used by the `VectorStore` service to perform a similarity search against the stored embeddings in PgVector.
    -   The most relevant text chunks (contexts) are retrieved based on cosine similarity or other distance metrics.

3.  **Chat Phase (Generation)**:
    -   The retrieved contexts, along with the user's current message and potentially the conversation history, are compiled into a comprehensive prompt.
    -   This prompt is sent to the Gemini AI's generative model.
    -   Gemini AI generates a coherent and contextually relevant answer based on the provided information.
    -   The `Chatbot` service returns this answer to the user and updates the conversation history.

---
*This README provides a comprehensive guide to understanding, setting up, and using the RAG Chatbot project.*