import { config } from './lib/config.js';
import { logger } from './lib/logger.js';
import { getMongo, collections } from './lib/mongo.js';
import { trimTokens } from './lib/tokenUtils.js';
import { uploadFile, createBatch, batchStatus, getRecentEmbeddingRequests } from './lib/openai.js';
import { RateLimiter } from './lib/rateLimiter.js';
import { ProgressTracker } from './lib/progressTracker.js';

const MAX_EMBEDDINGS_PER_BATCH = 50_000; // OpenAI embeddings request limit per batch
const SAFETY_WINDOW = 40_000; // ms before deadline to exit the function
const LOG_EVERY = 1000; // progress log cadence
const MAX_TOKENS_PER_DAY = 3_000_000; // 3 million tokens per day for tier 1

/**
 * DigitalOcean Functions entry‑point – Script‑1 (manual trigger)
 */

export async function main(payload = {}, ctx = {}) {
  const started = Date.now();
  const hardDeadline = started + 850_000; // 850 s (leave 50 s spare)

  const rateLimiter = new RateLimiter(config.openAiRateLimit);
  const progress = new ProgressTracker(config.processId);

  logger.info({ processId: config.processId }, 'sendForEmbed started');
  const srcParam = 'www.theatrenational.be';

  try {
    const client = await getMongo();
    const db = client.db(config.databaseName);
    const col = collections(db);

    /* ------------------------------------------------------------------
     * Optional one‑shot cleanup
     * Trigger with payload = { "clean": true }
     * -----------------------------------------------------------------*/
    const shouldClean = payload?.clean === true;

    if (shouldClean) {
      logger.warn('Clean flag detected – purging previous embeddings …');
      await Promise.all([
        col.embIndex.deleteMany({}),
        col.ragReady.deleteMany({}),
        col.signal.deleteMany({}),
        col.temp.deleteMany({}),
      ]);
      logger.warn('Cleanup complete; continuing with fresh run.');
    }

    /* ── Skip set: only fully‑processed chunks (timestamp≠null) ─────── */
    const batchedSet = new Set(
      await col.temp.distinct('chunk_id', {
        chunk_id: { $regex: `^${srcParam}` },
        batch_id: { $ne: null },
      }),
    );
    const totalChunks = await col.chunks.find(srcParam ? { source: srcParam } : {}).count();

    logger.warn(
      { alreadyBatched: batchedSet.size, totalChunks },
      batchedSet.size
        ? 'Resuming: some chunks already batched – continuing where we left off'
        : 'No batched chunks found – starting fresh',
    );

    /* For progress metrics */
    progress.metrics.totalChunks = totalChunks;
    progress.logProgress();

    if (totalChunks === batchedSet.size) {
      logger.warn('sendForEmbed has already processed all chunks');
      return { statusCode: 200, body: 'sendForEmbed has already processed all chunks' };
    }

    const recentEmbeddingRequests = await getRecentEmbeddingRequests();
    // According to the spec, the max tokens per chunk is ~512
    const tokensCapacity = MAX_TOKENS_PER_DAY - recentEmbeddingRequests * 512;

    if (tokensCapacity <= 0) {
      logger.warn('sendForEmbed has reached the max tokens per day limit');
      return { statusCode: 200, body: 'sendForEmbed has reached the max tokens per day limit' };
    }

    /* Stream chunks in deterministic order */
    const cursor = col.chunks.find(srcParam ? { source: srcParam } : {}, {
      projection: {
        html_content: 1,
        chunk_id: 1,
        page_counter: 1,
        source: 1,
        metadata: 1,
      },
    });

    const reqRows = [];
    const chunkIds = [];
    let embeddingsCount = 0;
    let batchTokens = 0;

    while (await cursor.hasNext()) {
      const doc = await cursor.next();
      const vectorId = `${doc.source}__p${doc.page_counter}__c${doc.chunk_id}`;

      if (batchedSet.has(vectorId)) {
        progress.incrementMetric('skippedChunks');
        continue;
      }

      // According to the spec, the max tokens per chunk is ~512
      const { text, tokens } = trimTokens(doc.html_content, 512);

      const nearTimeout = Date.now() >= hardDeadline - SAFETY_WINDOW;
      if (
        embeddingsCount + 1 > MAX_EMBEDDINGS_PER_BATCH ||
        batchTokens + tokens > tokensCapacity ||
        nearTimeout
      ) {
        const status = await flushBatch({ reqRows, chunkIds, col, rateLimiter, progress });
        // If we hit max tokens limit that can be processed simultaneously, exit and try another time.
        if (status === 'failed') {
          return {
            statusCode: 200,
            body: 'Creating batch failed: likely max enqueued tokens limit reached',
          };
        }
        if (nearTimeout) {
          return { statusCode: 200, body: 'sendForEmbed exited due to timeout' };
        }
        if (batchTokens + tokens > tokensCapacity) {
          return {
            statusCode: 200,
            body: `${progress.metrics.totalTokens} have been sent for processing in ${progress.metrics.totalBatches} batches before sendForEmbed has reached the max tokens per day limit`,
          };
        }
        reqRows.length = 0;
        chunkIds.length = 0;
        embeddingsCount = 0;
        batchTokens = 0;
      }

      reqRows.push(
        JSON.stringify({
          custom_id: vectorId,
          method: 'POST',
          url: '/v1/embeddings',
          body: {
            model: config.embeddingModel,
            input: text,
          },
        }),
      );
      embeddingsCount += 1;
      batchTokens += tokens;
      chunkIds.push(vectorId);

      // Track tokens & metrics
      batchTokens += tokens;
      progress.updateTokens(tokens);
      progress.incrementMetric('processedChunks');
      if (progress.metrics.processedChunks % LOG_EVERY === 0) {
        logger.warn(
          {
            processed: progress.metrics.processedChunks,
            skipped: progress.metrics.skippedChunks,
          },
          'progress',
        );
      }
    }

    const status = await flushBatch({ reqRows, chunkIds, col, rateLimiter, progress });
    if (status === 'failed') {
      return {
        statusCode: 200,
        body: 'Creating batch failed: likely max enqueued tokens limit reached',
      };
    }

    const duration = ((Date.now() - started) / 1000).toFixed(1) + 's';
    logger.info({ duration, metrics: progress.metrics }, 'sendForEmbed finished');

    return { statusCode: 200, body: JSON.stringify(progress.metrics) };
  } catch (err) {
    logger.error({ err }, 'sendForEmbed failed');
    return { statusCode: 500, body: err.message };
  }
}

async function flushBatch({ reqRows, chunkIds, col, rateLimiter, progress }) {
  if (!reqRows.length) return;

  // 1️⃣ increment batch counter
  progress.incrementMetric('totalBatches');

  try {
    await rateLimiter.waitForSlot();
    const jsonl = reqRows.join('\n');
    const buffer = Buffer.from(jsonl, 'utf8');
    const fileId = await uploadFile(buffer);
    const batch = await createBatch(fileId);

    // Check the status of the batch to make sure the batch has not failed
    // we are not assigning failed batch to the chunks.
    await new Promise(resolve => setTimeout(resolve, 30000));

    const statusResponse = await batchStatus(batch.id);

    if (statusResponse.status === 'failed') {
      logger.warn({ batchId: batch.id, status: statusResponse.status }, 'Batch failed');
      return 'failed';
    }

    const bulkOps = chunkIds.map(chunkId => ({
      updateOne: {
        filter: { chunk_id: chunkId },
        update: {
          $setOnInsert: {
            chunk_id: chunkId,
            batch_id: batch.id,
            timestamp: null,
          },
        },
        upsert: true,
      },
    }));
    await col.temp.bulkWrite(bulkOps);
    progress.incrementMetric('completedBatches');
    logger.info({ batchId: batch.id, queued: chunkIds.length }, 'Batch queued');

    return statusResponse.status;
  } catch (err) {
    progress.incrementMetric('failedBatches');
    logger.error({ err }, 'Batch creation failed');
    throw err;
  }
}

// global handlers
process.on('unhandledRejection', reason => {
  logger.error({ reason }, 'Unhandled promise rejection');
});
process.on('uncaughtException', err => {
  logger.fatal({ err }, 'Uncaught exception – shutting down');
  process.exit(1);
});
