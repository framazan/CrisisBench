# Langfuse Setup Guide

The CrisisBench repository uses Langfuse to track evaluation traces, run LLM-as-a-judge scorers, and visualize metrics.

You can use either **Langfuse Cloud** (recommended for ease of use) or **Self-Host Langfuse** via Docker.

## Option 1: Langfuse Cloud (Recommended)
1. Sign up for a free account at [Langfuse Cloud](https://cloud.langfuse.com).
2. Create a new Project (e.g., "CrisisBench Evals").
3. Go to **Settings** -> **API Keys** and generate a new set of API keys.
4. Export the keys to your terminal environment:
   ```bash
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
   ```

## Option 2: Self-Hosted Langfuse (Docker)
If you prefer to host your trace data locally, you can run Langfuse via Docker.

1. Clone the Langfuse repository and start the containers:
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```
2. Navigate to `http://localhost:3000` and create an admin account.
3. Create a Project and generate API keys.
4. Export the keys to your terminal environment:
   ```bash
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_BASE_URL="http://localhost:3000"
   ```

## Dashboard Creation
Our pipeline scripts (`run_pipeline.py`) can automatically generate analytics dashboards for you if you provide your Langfuse admin credentials. 

To enable automated dashboard creation, you must provide your account email and password when prompted by the CLI, or export them into your environment:
```bash
export LANGFUSE_ADMIN_EMAIL="admin@example.com"
export LANGFUSE_ADMIN_PASSWORD="yourpassword"
```

## Additional Environment Variables
Our pipelines also require your standard OpenAI API key for generating text and running the LLM judges:
```bash
export OPENAI_API_KEY="sk-proj-..."
```
