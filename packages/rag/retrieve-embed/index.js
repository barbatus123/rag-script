import { config }                  from './lib/config.js';
import { logger }                  from './lib/logger.js';
import { getMongo, collections }   from './lib/mongo.js';
import { batchStatus, downloadFile } from './lib/openai.js';
import { getWeaviate, mapClass }   from './lib/weaviate.js';
import { pipeline }                from 'stream/promises';
import split2                      from 'split2';

const SAFETY_WINDOW = 10_000;   // ms before the 850s timeout

export async function main(payload = {}, ctx = {}) {
    const start        = Date.now();
    const hardDeadline = start + 850_000;
    logger.info({ processId: config.processId }, 'retrieveEmbed started');

    try {
        // ── Init DB & Weaviate ─────────────────────────────────────
        const client = await getMongo();
        const db     = client.db(config.databaseName);
        const col    = collections(db);
        const wv     = getWeaviate();

        // ── Find all batches still pending (timestamp === null) ────
        const pendingBatches = await col.embIndex.distinct(
            'batch_id',
            { timestamp: null }
        );
        logger.warn({ pendingCount: pendingBatches.length }, 'pending batches found');

        if (pendingBatches.length === 0) {
            return { statusCode: 200, body: 'No pending batches' };
        }

        let processedBatches = 0;

        for (const batchId of pendingBatches) {
            // bail early if we’re almost out of time
            if (Date.now() >= hardDeadline - SAFETY_WINDOW) {
                logger.warn('Approaching timeout - exiting loop early');
                break;
            }

            // fetch status
            let status;
            try {
                status = await batchStatus(batchId);
            } catch (err) {
                logger.error({ err, batchId }, 'batchStatus failed');
                continue;
            }
            if (status.status !== 'completed') {
                logger.warn({ batchId, status: status.status }, 'batch not ready');
                continue;
            }

            // download JSONL stream
            let outStream;
            try {
                outStream = await downloadFile(status.output_file_id);
            } catch (err) {
                logger.error({ err, batchId }, 'downloadFile failed');
                continue;
            }

            logger.warn({ batchId }, 'processing completed batch');

            // collect all sources touched in this batch
            const processedSources = new Set();

            // ── Stream, insert into Weaviate + AI_FOR_RAG ───────────
            await pipeline(
                outStream,
                split2(JSON.parse),
                async function (source) {
                    for await (const row of source) {
                        const vectorId = row.id;
                        const embedding= row.embedding;

                        // parse vectorId → { source, page_counter, chunk_id }
                        const [src, rest]      = vectorId.split('__p');
                        const [pageStr, cid]   = rest.split('__c');
                        const page_counter     = parseInt(pageStr, 10);
                        const chunk_id         = parseInt(cid, 10);

                        const chunk = await col.chunks.findOne({
                            source, page_counter, chunk_id
                        }.source ? { source: src, page_counter, chunk_id } : {});
                        if (!chunk) {
                            logger.warn({ vectorId }, 'chunk not found');
                            continue;
                        }

                        processedSources.add(src);

                        const className = mapClass(chunk.metadata.data_type);

                        // upsert into Weaviate
                        try {
                            await wv.data
                                .creator()
                                .withClassName(className)
                                .withId(vectorId)
                                .withProperties({
                                    chunk_id:             chunk.chunk_id,
                                    page_counter:         chunk.page_counter,
                                    source:               chunk.source,
                                    rag_timestamp:        new Date().toISOString(),
                                    html_content:         chunk.html_content,
                                    metadata_data_type:   chunk.metadata.data_type,
                                    metadata_timestamp:   chunk.metadata.timestamp,
                                    metadata_crawl_depth: chunk.metadata.crawl_depth,
                                    metadata_title:       chunk.metadata.title
                                })
                                .withVector(embedding)
                                .do();
                        } catch (err) {
                            logger.error({ err, vectorId }, 'Weaviate insert failed');
                            continue;
                        }

                        // record in AI_FOR_RAG
                        await col.ragReady.updateOne(
                            { chunk_id: vectorId },
                            { $set: { timestamp: new Date() } },
                            { upsert: true }
                        );
                    }
                }
            );

            // ── Mark this batch done in AI_EMBEDDING ───────────────
            await col.embIndex.updateMany(
                { batch_id: batchId },
                { $set: { timestamp: new Date() } }
            );

            // ── SIGNAL_RAG: for each source, if all chunks are done, upsert one doc ─
            for (const src of processedSources) {
                const totalChunks = await col.chunks.countDocuments({ source: src });
                const doneChunks  = await col.ragReady.countDocuments({
                    chunk_id: { $regex: `^${src}__` }
                });
                if (totalChunks === doneChunks) {
                    await col.signal.updateOne(
                        { source: src, collection: mapClass(/* pick any data_type if uniform */) },
                        { $setOnInsert: { timestamp: null } },
                        { upsert: true }
                    );
                    logger.info({ source: src }, 'SIGNAL_RAG doc upserted');
                }
            }

            processedBatches++;
            logger.warn({ batchId, processedBatches }, 'batch processed fully');
        }

        const duration = ((Date.now() - start) / 1000).toFixed(1) + 's';
        logger.warn({ processedBatches, duration }, 'retrieveEmbed finished');
        return {
            statusCode: 200,
            body: `Processed ${processedBatches} batch(es) in ${duration}`
        };

    } catch (err) {
        logger.error({ err }, 'retrieveEmbed fatal error');
        return { statusCode: 500, body: err.message };
    }
}

process.on('unhandledRejection', reason => {
    logger.error({ reason }, 'Unhandled promise rejection');
});

process.on('uncaughtException', err => {
    logger.fatal({ err }, 'Uncaught exception – shutting down');
    process.exit(1);
});