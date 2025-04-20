import { config } from './lib/config.js';
import { logger } from './lib/logger.js';
import { getMongo, collections } from './lib/mongo.js';
import { countTokens, trimTokens } from './lib/tokenUtils.js';
import { uploadFile, createBatch } from './lib/openai.js';
import { RateLimiter } from './lib/rateLimiter.js';
import { ProgressTracker } from './lib/progressTracker.js';

const MAX_EMBEDDINGS_PER_BATCH = 50_000;    // OpenAI hard limit per batch
const SAFETY_WINDOW  = 20_000;    // ms before hardDeadline to exit loop
const LOG_EVERY      = 1000;       // progress log cadence
const BATCH_PER_REQUEST  = 16;

/**
 * DigitalOcean Functions entry‑point – Script‑1 (manual trigger)
 */

export async function main(payload = {}, ctx = {}) {
    const started      = Date.now();
    const hardDeadline = started + 850_000;          // 850 s (leave 50 s spare)

    const rateLimiter  = new RateLimiter(config.openAiRateLimit);
    const progress     = new ProgressTracker(config.processId);

    logger.info({ processId: config.processId }, 'sendForEmbed started');

    try {
        const client = await getMongo();
        const db     = client.db(config.databaseName);
        const col    = collections(db);

        /* ------------------------------------------------------------------
        * Optional one‑shot cleanup
        * Trigger with payload = { "clean": true }
        * -----------------------------------------------------------------*/
        const shouldClean = payload?.clean === true;

        if (shouldClean) {
            logger.warn('🧹 Clean flag detected – purging previous embeddings …');
            await Promise.all([
                col.embIndex.deleteMany({}),
                col.ragReady.deleteMany({}),
                col.signal.deleteMany({}),
                col.temp.deleteMany({})
            ]);
            logger.warn('Cleanup complete; continuing with fresh run.');
        }

        /* ── Skip set: only fully‑processed chunks (timestamp≠null) ─────── */
        const completedSet = new Set(
            await col.embIndex.distinct('chunk_id', { timestamp: { $ne: null } })
        );

        logger.warn(
            {
               alreadyFinished: completedSet.size,
               totalChunks: progress.metrics.totalChunks
            },
         completedSet.size
           ? 'Resuming: some chunks already complete – continuing where we left off'
               : 'No completed chunks found – starting fresh'
        );

        /* For progress metrics */
        progress.metrics.totalChunks = await col.chunks.countDocuments();
        progress.logProgress();

        /* Stream chunks in deterministic order */
        const cursor = col.chunks
            .find({}, {
                projection: {
                    html_content: 1,
                    chunk_id: 1,
                    page_counter: 1,
                    source: 1,
                    metadata: 1
                }
            });

        const jsonlRows = [];
        const reqRows = [];
        const chunkIds  = [];
        let embeddingsCount = 0;

        const flushEmbeddingsRequest = () => {
            if (reqRows.length === 0) return;

            jsonlRows.push(
                JSON.stringify({
                    custom_id: reqRows.map(row => row.id).join(','),
                    method: 'POST',
                    url: '/v1/embeddings',
                    body: {
                        model: config.embeddingModel,
                        input: reqRows.map(row => row.input)
                    }
                })
            );
        }

        while (await cursor.hasNext()) {
            const doc = await cursor.next();
            const vectorId = `${doc.source}__p${doc.page_counter}__c${doc.chunk_id}`;

            if (completedSet.has(vectorId)) {
                progress.incrementMetric('skippedChunks');
                continue;
            }

            const { text, tokens } = trimTokens(doc.html_content);
            reqRows.push({
                id:    vectorId,
                input: text
            });
            embeddingsCount += 1;
            chunkIds.push(vectorId);

            if (reqRows.length === BATCH_PER_REQUEST) {
                flushEmbeddingsRequest();
                reqRows.length = 0;
            }

            progress.updateTokens(tokens);
            progress.incrementMetric('processedChunks');
            if (progress.metrics.processedChunks % LOG_EVERY === 0) {
                logger.info({ processed: progress.metrics.processedChunks }, 'progress');
            }

            const nearTimeout = Date.now() >= hardDeadline - SAFETY_WINDOW;
            if (embeddingsCount >= MAX_EMBEDDINGS_PER_BATCH || nearTimeout) {
                flushEmbeddingsRequest();
                await flushBatch({ jsonlRows, chunkIds, col, rateLimiter, progress });
                jsonlRows.length = 0;
                chunkIds.length  = 0;
                reqRows.length = 0;
                embeddingsCount = 0;
                if (nearTimeout) break;      // exit loop safely
            }

            if (progress.metrics.processedChunks % LOG_EVERY === 0) {
                logger.info(
                    {
                        processed: progress.metrics.processedChunks,
                        skipped:   progress.metrics.skippedChunks,
                        alreadyFinished: completedSet.size
                    },
                    'in‑progress'
                );
            }
        }

        flushEmbeddingsRequest();
        await flushBatch({ jsonlRows, chunkIds, col, rateLimiter, progress });

        const duration = ((Date.now() - started) / 1000).toFixed(1) + 's';
        logger.info({ duration, metrics: progress.metrics },
        completedSet.size
           ? 'sendForEmbed finished (resume run)'
            : 'sendForEmbed finished (initial run)');

        return { statusCode: 200, body: JSON.stringify(progress.metrics) };
    } catch (err) {
        logger.error({ err }, 'sendForEmbed failed');
        return { statusCode: 500, body: err.message };
    }
}

/* ──────────────────────────────────────────────────────────────────────────
   Flush the current batch

   1. Upsert placeholder docs `{ timestamp: null }` so restarts can resume.
   2. Create the OpenAI batch (rate‑limited).
   3. Back‑fill `batch_id` for those same docs.
─────────────────────────────────────────────────────────────────────────── */
async function flushBatch({ jsonlRows, chunkIds, col, rateLimiter, progress }) {
    if (!jsonlRows.length) return;

    progress.incrementMetric('totalBatches');

    const bulkOps = chunkIds.map(chunkId => ({
        updateOne: {
            filter: { chunk_id: chunkId },
            update: { 
                $setOnInsert: { 
                    chunk_id: chunkId,
                    batch_id: null, 
                    timestamp: null 
                }
            },
            upsert: true
        }
    }));
    await col.embIndex.bulkWrite(bulkOps);

    try {
        await rateLimiter.waitForSlot();
        const jsonl = jsonlRows.join('\n');
        const buffer = Buffer.from(jsonl, 'utf8');
        const fileId = await uploadFile(buffer);
        const batch  = await createBatch(fileId);

        await col.embIndex.updateMany(
            { chunk_id: { $in: chunkIds } },
            { $set: { batch_id: batch.id } }
        );

        progress.incrementMetric('completedBatches');
        logger.info({ batchId: batch.id, queued: chunkIds.length }, 'Batch queued');
    } catch (err) {
        progress.incrementMetric('failedBatches');
        logger.error({ err }, 'Batch creation failed');
        throw err;
    }
}

process.on('unhandledRejection', reason => {
    logger.error({ reason }, 'Unhandled promise rejection');
});

process.on('uncaughtException', err => {
    logger.fatal({ err }, 'Uncaught exception – shutting down');
    process.exit(1);
});
