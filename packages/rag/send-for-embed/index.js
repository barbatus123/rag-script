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
    const start        = Date.now();
    const hardDeadline = start + 850_000;
    const rateLimiter  = new RateLimiter(config.openAiRateLimit);
    const progress     = new ProgressTracker(config.processId);

    logger.info({ processId: config.processId }, 'sendForEmbed started');

    try {
        // ── Setup Mongo & Collections ───────────────────────────────────
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
            logger.warn('Clean flag detected – purging previous embeddings …');
            await Promise.all([
                col.embIndex.deleteMany({}),
                col.ragReady.deleteMany({}),
                col.signal.deleteMany({}),
                col.temp.deleteMany({})
            ]);
            logger.warn('Cleanup complete; continuing with fresh run.');
        }

        // ── Build skip‑set (only fully‑completed chunks) ────────────────
        const completedSet = new Set(
            await col.embIndex.distinct('chunk_id', { timestamp: { $ne: null } })
        );
        progress.metrics.totalChunks = await col.chunks.countDocuments();

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

        let jsonlRows     = [];
        let chunkIds      = [];
        let batchTokens   = 0;
        // Load initial enqueued tokens (sum of token_count for pending chunks)
        const agg = await col.embIndex.aggregate([
            {$match: {timestamp: null}},
            {$group: {_id: null, total: {$sum: '$token_count'}}}
        ]).toArray();
        let enqueuedTokens = agg.length > 0 ? agg[0].total : 0;

        while (await cursor.hasNext()) {
            const doc = await cursor.next();
            const vectorId = `${doc.source}__p${doc.page_counter}__c${doc.chunk_id}`;

            // skip already done
            if (completedSet.has(vectorId)) {
                progress.incrementMetric('skippedChunks');
                continue;
            }

            // enforce per‑chunk token limit only
            const tokens = countTokens(doc.html_content);
            if (tokens > config.maxTokenPerInput) {
                logger.warn({vectorId, tokens}, 'Chunk too large – skipped');
                progress.incrementMetric('skippedChunks');
                continue;
            }

            // decide if we need to flush before adding this chunk:
            // 1) row‐count cap
            // 2) per‐chunk too‐big guard is earlier
            // 3) org‐level enqueued‐token cap
            // 4) nearing timeout
            const remainingOrg = config.orgTokenLimit - enqueuedTokens;
            const needFlush =
                jsonlRows.length > 0 && (
                    jsonlRows.length >= config.batchSize ||
                    batchTokens + tokens > remainingOrg ||
                    Date.now() >= hardDeadline - SAFETY_WINDOW
                );

            if (needFlush) {
                await flushBatch({ jsonlRows, chunkIds, batchTokens, col, rateLimiter, progress });
                // update our in‐memory enqueued count
                enqueuedTokens += batchTokens;

                jsonlRows   = [];
                chunkIds    = [];
                batchTokens = 0;
                if (Date.now() >= hardDeadline - SAFETY_WINDOW) {
                    logger.warn('Safety window reached – exiting loop');
                    break;
                }
            }

            // append the JSONL line
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
                logger.info(
                    {
                        processed: progress.metrics.processedChunks,
                        batchSize: jsonlRows.length,
                        batchTokens
                    },
                    'progress'
                );
            }
        }

        // final flush if anything remains
        if (jsonlRows.length) {
            await flushBatch({ jsonlRows, chunkIds, batchTokens, col, rateLimiter, progress });
        }

        const duration = ((Date.now() - start)/1000).toFixed(1) + 's';
        logger.info({ duration, metrics: progress.metrics }, 'sendForEmbed finished');
        return { statusCode: 200, body: JSON.stringify(progress.metrics) };

    } catch (err) {
        logger.error({ err }, 'sendForEmbed failed');
        return { statusCode: 500, body: err.message };
    }
}


/**
 * Flush the current JSONL batch:
 * 1) Mark docs in AI_EMBEDDING
 * 2) Upload file & create OpenAI batch
 * 3) Back‑fill batch_id in AI_EMBEDDING
 */
async function flushBatch({ jsonlRows, chunkIds, batchTokens, col, rateLimiter, progress }) {
    if (!jsonlRows.length) return;

    // 1️⃣ increment batch counter
    progress.incrementMetric('totalBatches');

    // compute an average token_count per chunk
    const avgTokenCount = batchTokens / chunkIds.length;

    // 2️⃣ mark chunks as in‐flight and record token_count
    await col.embIndex.bulkWrite(
        chunkIds.map(id => ({
            updateOne: {
                filter: { chunk_id: id },
                update: {
                    $setOnInsert: {
                        batch_id:    null,
                        timestamp:   null,
                        token_count: avgTokenCount
                    }
                },
                upsert: true
            }
        })),
        { ordered: false }
    );

    try {
        // 3️⃣ rate‐limit and upload the file
        await rateLimiter.waitForSlot();
        const payload = Buffer.from(jsonlRows.join('\n'), 'utf8');
        const fileId  = await uploadFile(payload);

        // 4️⃣ create the OpenAI batch
        const batch   = await createBatch(fileId);

        // 5️⃣ back‐fill batch_id for these chunks
        await col.embIndex.updateMany(
            { chunk_id: { $in: chunkIds } },
            { $set: { batch_id: batch.id } }
        );

        progress.incrementMetric('completedBatches');
        logger.info(
            { batchId: batch.id, count: chunkIds.length },
            'Batch queued'
        );
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