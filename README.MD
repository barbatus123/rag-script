### Deployment

Add .env file with required environment variables and run:

```bash
doctl serverless deploy .
```

To view rag/send-for-embed or rag/retrieve-embed last logs
```bash
doctl serverless activations logs --function rag/send-for-embed --follow
```
