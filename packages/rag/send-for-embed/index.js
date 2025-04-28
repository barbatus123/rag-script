import { config } from './lib/config.js';
import { logger } from './lib/logger.js';
import { getMongo, collections } from './lib/mongo.js';
import { trimTokens } from './lib/tokenUtils.js';
import { uploadFile, createBatch, batchStatus } from './lib/openai.js';
import { RateLimiter } from './lib/rateLimiter.js';
import { ProgressTracker } from './lib/progressTracker.js';

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const SAFETY_WINDOW = 40_000; // ms before deadline to exit the function
const LOG_EVERY = 1000; // progress log cadence

async function getRecentlyBatchedTokens(col, offsetMins = 0) {
  const oneDayAgo = new Date(Date.now() - (24 * 3_600_000 - offsetMins * 60_000));

  const result = await col.embIndex
    .aggregate([
      {
        $match: {
          batched_at: { $gte: oneDayAgo },
        },
      },
      {
        $group: {
          _id: null,
          totalTokens: { $sum: '$tokens' },
        },
      },
    ])
    .toArray();

  return result[0]?.totalTokens || 0;
}

/**
 * DigitalOcean Functions entry‑point – Script‑1 (manual trigger)
 */

export async function main(payload = {}, ctx = {}) {
  const started = Date.now();
  const hardDeadline = started + 850_000; // 850 s (leave 50 s spare)

  const rateLimiter = new RateLimiter(config.openAiRateLimit);
  const progress = new ProgressTracker(config.processId);

  logger.info({ processId: config.processId }, 'sendForEmbed started');
  // Source param, if provided, will only process chunks for that source
  const srcParam = payload.source;

  try {
    const client = await getMongo();
    const db = client.db(config.databaseName);
    const col = collections(db);

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

    const doneSources = await col.signal.distinct('source');

    const batchQuery = {
      batch_id: { $ne: null },
      chunk_id: doneSources
        ? { $not: { $regex: `^${doneSources.map(escapeRegExp).join('|')}` } }
        : {},
    };
    const batchedSet = new Set(
      await col.embIndex.distinct(
        'chunk_id',
        srcParam
          ? {
              ...batchQuery,
              chunk_id: { $regex: `^${escapeRegExp(srcParam)}` },
            }
          : batchQuery,
        { projection: { chunk_id: 1 } },
      ),
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

    const recentTokens = await getRecentlyBatchedTokens(col, 10);
    logger.warn({ consumedTokens: recentTokens }, 'Tokens consumed by recent batches');
    const tokensCapacity = config.orgTokenLimit - recentTokens;

    if (tokensCapacity <= 0) {
      logger.warn('sendForEmbed has reached the max tokens per day limit');
      return { statusCode: 200, body: 'sendForEmbed has reached the max tokens per day limit' };
    }

    /* Stream chunks in deterministic order */
    const query = {
      $or: [{ batch_id: null }, { batch_id: { $exists: false } }],
      source: doneSources ? { $not: { $in: doneSources } } : {},
    };
    const cursor = col.chunks.find(
      srcParam
        ? {
            ...query,
            source: srcParam,
          }
        : query,
      {
        projection: {
          html_content: 1,
          chunk_id: 1,
          page_counter: 1,
          source: 1,
          metadata: 1,
        },
      },
    );

    const reqRows = [];
    const chunks = [];
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
        embeddingsCount + 1 > config.maxEmbeddingsPerBatch ||
        batchTokens + tokens > tokensCapacity ||
        nearTimeout
      ) {
        const status = await flushBatch({ reqRows, chunks, col, rateLimiter, progress });
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
        chunks.length = 0;
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
      chunks.push({
        vectorId,
        tokens,
      });

      // Track tokens & metrics
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

    const status = await flushBatch({ reqRows, chunks, col, rateLimiter, progress });
    if (status === 'failed') {
      return {
        statusCode: 200,
        body: 'Creating batch failed: likely max enqueued tokens limit reached',
      };
    }

    const duration = ((Date.now() - started) / 1000).toFixed(1) + 's';
    logger.warn({ duration, metrics: progress.metrics }, 'sendForEmbed finished');

    return { statusCode: 200, body: JSON.stringify(progress.metrics) };
  } catch (err) {
    logger.error({ err }, 'sendForEmbed failed');
    return { statusCode: 500, body: err.message };
  }
}

async function flushBatch({ reqRows, chunks, col, rateLimiter, progress }) {
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

    const bulkOps = chunks.map(({ vectorId, tokens }) => ({
      updateOne: {
        filter: { chunk_id: vectorId },
        update: {
          $set: {
            batch_id: batch.id,
            timestamp: null,
            batched_at: new Date(),
            tokens,
          },
        },
        upsert: true,
      },
    }));
    await col.embIndex.bulkWrite(bulkOps);
    progress.incrementMetric('completedBatches');
    logger.warn({ batchId: batch.id, queued: chunks.length }, 'Batch queued');

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
