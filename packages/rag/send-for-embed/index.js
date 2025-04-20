import { config } from './lib/config.js';
import { logger } from './lib/logger.js';
import { getMongo, collections } from './lib/mongo.js';
import { countTokens } from './lib/tokenUtils.js';
import { uploadFile, createBatch } from './lib/openai.js';
import { RateLimiter } from './lib/rateLimiter.js';
import { ProgressTracker } from './lib/progressTracker.js';

const SAFETY_WINDOW  = 20_000;    // ms before hardDeadline to exit loop
const LOG_EVERY      = 1000;       // progress log cadence

/**
 * DigitalOcean Functions entry‑point – Script‑1 (manual trigger)
 */

export async function main(payload = {}, ctx = {}) {
    const started      = Date.now();
    const hardDeadline = started + 850_000;          // 850 s (leave 50 s spare)

    const rateLimiter  = new RateLimiter(config.openAiRateLimit);
    const progress     = new ProgressTracker(config.processId);

    logger.info({ processId: config.processId }, 'sendForEmbed started');

    try {
        /* ── Connect to Mongo ─────────────────────────────────────────────── */
        const client = await getMongo();
        const db     = client.db(config.databaseName);
        const col    = collections(db);

        /* ------------------------------------------------------------------
        * Optional one‑shot cleanup
        * Trigger with payload = { "clean": true }
        * Deletes:
        *   • AI_EMBEDDING        – all rows
        *   • AI_FOR_RAG          – tracking table
        *   • SIGNAL_RAG          – per‑source done flags
        *   • (optionally) AI_TEMP – any leftovers
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

        let jsonlRows = [];
        let chunkIds  = [];
        let batchTokens = 0;

        while (await cursor.hasNext()) {
            const doc = await cursor.next();
            const vectorId = `${doc.source}__p${doc.page_counter}__c${doc.chunk_id}`;

            if (completedSet.has(vectorId)) {
                progress.incrementMetric('skippedChunks');
                continue;
            }

            const tokens = countTokens(doc.html_content);
            if (tokens > config.maxTokenPerBatch) {
                logger.warn({vectorId, tokens}, 'Chunk too large – skipped');
                progress.incrementMetric('skippedChunks');
                continue;
            }

            // If adding this chunk would exceed either the token or count limits, flush first
            const needFlush =
                jsonlRows.length > 0 && (
                    jsonlRows.length >= Math.min(config.batchSize, config.maxJsonlRows) ||
                    batchTokens + tokens > config.maxTokenPerBatch ||
                    Date.now() >= hardDeadline - SAFETY_WINDOW
                );

            if (needFlush) {
                await flushBatch({jsonlRows, chunkIds, col, rateLimiter, progress});
                jsonlRows = [];
                chunkIds = [];
                batchTokens = 0;
                if (Date.now() >= hardDeadline - SAFETY_WINDOW) {
                    logger.warn('Safety window reached — exiting loop');
                    break;
                }
            }

            jsonlRows.push(JSON.stringify({
                custom_id: vectorId,
                method: 'POST',
                url: '/v1/embeddings',
                body: {
                    model: config.embeddingModel,
                    input: doc.html_content
                }
            }));
            chunkIds.push(vectorId);

            // Track tokens & metrics
            batchTokens += tokens;
            progress.updateTokens(tokens);
            progress.incrementMetric('processedChunks');
            if (progress.metrics.processedChunks % LOG_EVERY === 0) {
                logger.info({
                    processed: progress.metrics.processedChunks,
                    batchSize: jsonlRows.length,
                    batchTokens
                }, 'progress');
            }


            const nearTimeout = Date.now() >= hardDeadline - SAFETY_WINDOW;
            if (jsonlRows.length >= config.maxJsonlRows || nearTimeout) {
                await flushBatch({ jsonlRows, chunkIds, col, rateLimiter, progress });
                jsonlRows.length = 0;
                chunkIds.length  = 0;
                if (nearTimeout) break;      // exit loop safely
            }

            if (progress.metrics.processedChunks % LOG_EVERY === 0) {
                logger.warn(
                    {
                        processed: progress.metrics.processedChunks,
                        skipped:   progress.metrics.skippedChunks,
                        alreadyFinished: completedSet.size
                    },
                    'in‑progress'
                );
            }
        }

        /* final flush */
        if (jsonlRows.length) {
            await flushBatch({ jsonlRows, chunkIds, col, rateLimiter, progress });
        }

        const duration = ((Date.now() - started) / 1000).toFixed(1) + 's';
        logger.warn({ duration, metrics: progress.metrics },
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

    /* 1️⃣  Mark chunks as in‑flight */
    await col.embIndex.bulkWrite(
        chunkIds.map(id => ({
            updateOne: {
                filter:  { chunk_id: id },
                update:  { $setOnInsert: { batch_id: null, timestamp: null } },
                upsert:  true
            }
        })),
        { ordered: false }
    );



    try {
        /* 2️⃣  Send to OpenAI */
        await rateLimiter.waitForSlot();
        const jsonl = jsonlRows.join('\n');
        const buffer = Buffer.from(jsonl, 'utf8');        // ← always a Buffer
        const fileId = await uploadFile(buffer);
        const batch  = await createBatch(fileId);

        /* 3️⃣  Update batch_id */
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