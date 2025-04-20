import axios       from 'axios';
import axiosRetry  from 'axios-retry';
import FormData    from 'form-data';
import { config }  from './config.js';

/* ------------------------------------------------------------------ */
/*  Axios client with automatic retry on 429 + 5xx                    */
/* ------------------------------------------------------------------ */
const client = axios.create({
    baseURL: config.openAiBaseUrl || 'https://api.openai.com/v1',
    headers:  { Authorization: `Bearer ${config.openAiApiKey}` },
    timeout: 30_000
});

axiosRetry(client, {
    retries: 5,
    retryDelay: axiosRetry.exponentialDelay,
    retryCondition: err =>
        !err.response ||                     // network error
        err.response.status >= 500 ||        // 5xx
        err.response.status === 429          // rate‑limited
});

/* ------------------------------------------------------------------ */
/*  Upload a JSONL file for the /batches endpoint.                     */
/*  Accepts Buffer or string, **never** Blob/Headers objects.          */
/* ------------------------------------------------------------------ */
export async function uploadFile(content) {
    if (!Buffer.isBuffer(content) && typeof content !== 'string') {
        throw new TypeError(
            'uploadFile() expects a Buffer or UTF‑8 string; received ' +
            Object.prototype.toString.call(content)
        );
    }
    const buffer = Buffer.isBuffer(content)
        ? content
        : Buffer.from(content, 'utf8');

    const form = new FormData();
    form.append('purpose', 'batch');
    form.append('file', buffer, {
        filename:   'input.jsonl',
        contentType:'application/json'
    });

    // compute Content-Length
    const length = await new Promise((resolve, reject) => {
        form.getLength((err, len) => err ? reject(err) : resolve(len));
    });

    const headers = {
        ...form.getHeaders(),         // includes multipart boundary
        'Content-Length': length      // ensure the server knows payload size
    };

    const { data } = await client.post(
        '/files',
        form,
        { headers }
    );
    return data.id;
}

/* ------------------------------------------------------------------ */
/*  One‑shot synchronous embeddings (rarely used in our pipeline)      */
/* ------------------------------------------------------------------ */
export async function embedBatch(inputs) {
    const { data } = await client.post('/embeddings', {
        model: config.embeddingModel,
        input: inputs
    });
    return data.data;                  // [{index, embedding, object}]
}

/* ------------------------------------------------------------------ */
/*  Batch lifecycle helpers                                            */
/* ------------------------------------------------------------------ */
export async function createBatch(fileId) {
    const { data } = await client.post('/batches', {
        input_file_id: fileId,
        endpoint: '/v1/embeddings',
        completion_window: '24h'
    });
    return data;                       // {id,status,...}
}

export async function batchStatus(batchId) {
    const { data } = await client.get(`/batches/${batchId}`);
    return data;
}

export async function downloadFile(fileId) {
    const { data } = await client.get(`/files/${fileId}/content`, {
        responseType: 'stream'
    });
    return data;                       // JSONL stream
}