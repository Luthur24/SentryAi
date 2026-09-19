SentryAI

SentryAI is a unified AI platform that provides developers with a single API for advanced language models, web-powered intelligence, tools, and multimodal AI capabilities.

Overview

SentryAI is designed to simplify AI application development by providing a unified interface for interacting with multiple model capabilities through a single backend.

Instead of requiring an application to implement separate integrations for different AI providers, SentryAI centralizes model access, routing, fallback handling, web research, tool execution, and response processing behind a consistent API.

The system is built around a provider abstraction layer that allows model requests to be routed across available AI backends while keeping provider-specific implementation details inside the platform.

Core Features

- Unified AI API
- Multi-provider model routing
- Automatic provider fallback
- Model fallback within providers
- Multiple API-key support
- Rate-limit handling
- Authentication and request validation
- Web search
- Web page fetching
- Search result injection into model context
- Tool execution loop
- Request tracing
- Configurable model routing order
- Configurable context limits
- Configurable output limits
- OpenAI-style chat-completion architecture
- Streaming-oriented backend architecture
- Multimodal AI support
- Environment-based configuration

AI Routing Architecture

SentryAI separates the client-facing API from the underlying model providers.

Developer Application
        │
        ▼
   SentryAI API
        │
        ▼
   Request Router
        │
        ├───────────────┐
        │               │
        ▼               ▼
   Model Route      Web Tools
        │               │
        ▼               ▼
 Provider Adapter   Web Search
        │           Page Fetch
        ▼
   AI Provider
        │
        ▼
    Response

The backend maintains a configurable provider order and attempts available routes sequentially when a provider or model cannot successfully answer a request.

Provider Abstraction

The current implementation supports multiple model backends through a common routing layer.

The backend maintains provider-specific model lists and API-key configurations while exposing a unified model-calling interface to the rest of the application.

This abstraction makes it possible to:

- Change provider priority
- Add or remove models
- Fall back between providers
- Handle provider-specific errors
- Rotate between multiple API keys
- Keep provider-specific request logic isolated

Failure Handling & Fallback

One of the main engineering goals of SentryAI is resilience against external provider failures.

The routing layer handles conditions including:

- Rate limits
- Authentication failures
- Model unavailability
- Server errors
- Empty responses
- Network failures

For example, a rate-limited model request can move to another configured key or model instead of immediately failing the entire request.

Request
   │
   ▼
Provider A
   │
   ├── Success ───────► Response
   │
   └── Failure
          │
          ▼
      Provider B
          │
          ├── Success ─────► Response
          │
          └── Failure
                 │
                 ▼
             Provider C
                 │
                 ▼
              Response

Web Intelligence

SentryAI can augment model requests with external web information.

The backend integrates web search and can:

- Search the web
- Retrieve search results
- Include images from search results
- Fetch pages
- Inject retrieved information into model context
- Perform additional searches during a request

The web-search layer is bounded by configurable limits to prevent uncontrolled tool usage.

Tool Execution

The backend supports an iterative tool-processing loop.

A model can determine that additional information is required, invoke an available tool, receive the result, and continue processing.

The implementation places explicit limits on tool iterations and search results.

User Request
     │
     ▼
   Model
     │
     ├── Direct Answer ─────────► Response
     │
     └── Needs Information
              │
              ▼
             Tool
              │
              ▼
         Tool Result
              │
              ▼
             Model
              │
              ▼
           Response

Context Management

SentryAI defines shared input and output token ceilings and provider-specific output limits.

The current implementation uses a configurable 120,000-token input ceiling and provider-specific output caps so that requests remain within the supported limits of the configured model routes.

Tracing

The backend includes request tracing for important stages of the routing process.

Tracing events can identify events such as:

- Route attempts
- Successful routes
- Web searches
- Search failures
- Tool processing

This provides visibility into how a request was processed internally.

Technology Stack

- Python
- Flask
- REST APIs
- AI model APIs
- Web search APIs
- HTTP/JSON
- Environment-based configuration
- Vercel
- Git & GitHub

Repository Structure

SentryAi/
├── server.py
├── index.html
├── requirements.txt
├── .vercelignore
└── README.md

The primary backend implementation is contained in "server.py", while "index.html" provides the web interface.

Running Locally

Clone the repository:

git clone <repository-url>
cd SentryAi

Install dependencies:

pip install -r requirements.txt

Configure the required environment variables for the AI providers, web search, application configuration, and deployment settings.

Start the backend:

python server.py

Environment Configuration

Provider credentials and application secrets should be supplied through environment variables.

Example:

GEMINI_API_KEY
GROQ_API_KEY_1
GROQ_API_KEY_2
MISTRAL_API_KEY_1
MISTRAL_API_KEY_2
ZAI_API_KEY_1
ZAI_API_KEY_2
TAVILY_API_KEY_1
TAVILY_API_KEY_2
PROVIDER_ORDER

Never commit real credentials to source control.

Security Notice

Before using this repository publicly or in a job application, rotate any credentials that have previously appeared in the source code.

The current public "server.py" contains provider credentials in configuration defaults. Those credentials should be revoked, replaced with environment variables, and removed from the repository history.

Project Status

AI Platform MVP / Active Development

SentryAI is an experimental unified AI infrastructure project focused on model abstraction, routing, resilience, web intelligence, and developer-facing AI APIs.