# Deploying Code Assistant to Google Cloud Run

This guide explains how to deploy the Code Assistant application to Google Cloud Run.

## Prerequisites

1. A Google Cloud Platform account with billing enabled
2. Google Cloud CLI (`gcloud`) installed and configured
3. API keys for your LLM providers (OpenAI, Anthropic)
4. Pinecone account and API key (or MongoDB Atlas if using MongoDB)

## Deployment Steps

### 1. Set up Secret Manager

Create secrets for your API keys in Google Cloud Secret Manager:

```bash
# Create secrets for API keys
gcloud secrets create openai-api-key --replication-policy="automatic"
gcloud secrets create anthropic-api-key --replication-policy="automatic"
gcloud secrets create pinecone-api-key --replication-policy="automatic" 
gcloud secrets create pinecone-index-name --replication-policy="automatic"

# Set the values for each secret
echo -n "your_openai_key" | gcloud secrets versions add openai-api-key --data-file=-
echo -n "your_anthropic_key" | gcloud secrets versions add anthropic-api-key --data-file=-
echo -n "your_pinecone_key" | gcloud secrets versions add pinecone-api-key --data-file=-
echo -n "your_index_name" | gcloud secrets versions add pinecone-index-name --data-file=-
```

### 2. Grant Access to Secrets

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')

# Grant access to the Cloud Run service account
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-cloudrun.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-cloudrun.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding pinecone-api-key \
  --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-cloudrun.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding pinecone-index-name \
  --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-cloudrun.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 3. Deploy with Cloud Build

```bash
# Submit the build
gcloud builds submit --config=cloudbuild.yaml
```

This will:
1. Build a Docker container with your application
2. Push it to Google Container Registry
3. Deploy it to Cloud Run with your secret environment variables

### 4. Access Your Deployed API

Once deployment completes, Cloud Build will output a URL for your service. You can also get it with:

```bash
gcloud run services describe code-assistant --platform managed --region us-central1 --format 'value(status.url)'
```

## Manual Deployment

You can also deploy manually without Cloud Build:

```bash
# Build the container
docker build -t gcr.io/your-project-id/code-assistant:latest .

# Push to Container Registry
docker push gcr.io/your-project-id/code-assistant:latest

# Deploy to Cloud Run
gcloud run deploy code-assistant \
  --image gcr.io/your-project-id/code-assistant:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest,PINECONE_INDEX_NAME=pinecone-index-name:latest
```

## Environment Variables

The application uses these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Your Pinecone index name
- `MONGODB_URI`: (Optional) Your MongoDB connection URI if using MongoDB
- `PORT`: The port to run the API server (defaults to 8080)
- `HOST`: The host to bind to (defaults to 0.0.0.0)

## Troubleshooting

If you encounter issues with deployment:

1. Check Cloud Build logs: `gcloud builds list`
2. Check Cloud Run logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=code-assistant"`
3. Test locally with Docker: `docker run -p 8080:8080 --env-file .env gcr.io/your-project-id/code-assistant:latest`