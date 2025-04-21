import { pipeline } from 'stream/promises';
import split2 from 'split2';
import { v4 as uuidv4 } from 'uuid';

import { config } from '../lib/config.js';
import { logger } from '../lib/logger.js';
import { getMongo, collections } from '../lib/mongo.js';
import { batchStatus, downloadFile } from '../lib/openai.js';
import { getWeaviate, mapClass } from '../lib/weaviate.js';

const SAFETY_WINDOW = 10_000; // ms before the 850s timeout

async function processFailedBatch({ col, fileId }) {
  const stream = await downloadFile(fileId);
  const failedVectorIds = [];
  await pipeline(stream, split2(JSON.parse), async function (source) {
    for await (const row of source) {
      failedVectorIds.push(row.custom_id);
    }
  });

  await col.embIndex.updateMany(
    { chunk_id: { $in: failedVectorIds } },
    { $set: { batch_id: null, batched_at: null } },
  );

  logger.info({ failedChunks: failedVectorIds.length }, 'Remove batch_id from failed chunks');
}

export async function main() {
  const start = Date.now();
  const hardDeadline = start + 850_000;
  logger.info({ processId: config.processId }, 'retrieveEmbed started');

  // collect all sources touched in this batch
  const processedSources = new Set();

  try {
    // ── Init DB & Weaviate ─────────────────────────────────────
    const client = await getMongo();
    const db = client.db(config.databaseName);
    const col = collections(db);
    const wv = getWeaviate();

    // ── Find pending chunks and batches to process ────
    const pendingChunks = await col.embIndex.find({ timestamp: null }).toArray();
    const pendingBatches = new Set(pendingChunks.map(chunk => chunk.batch_id));
    logger.warn({ pendingCount: pendingBatches.size }, 'Pending batches found');

    if (pendingBatches.size === 0) {
      return { statusCode: 200, body: 'No pending batches' };
    }

    let processedBatches = 0;
    for (const batchId of pendingBatches) {
      // bail early if we're almost out of time
      if (Date.now() >= hardDeadline - SAFETY_WINDOW) {
        logger.warn('Approaching timeout - exiting loop early');
        break;
      }

      // fetch status
      let status;
      try {
        status = await batchStatus(batchId);
      } catch (err) {
        logger.error({ err, batchId }, 'Getting batchStatus failed');
        continue;
      }

      if (status.status === 'failed') {
        logger.error({ batchId }, 'Batch failed');
        try {
          await processFailedBatch({ col, fileId: status.input_file_id });
        } catch (err) {
          logger.error({ err, batchId }, 'Failed to process failed batch');
        }
        continue;
      }

      if (status.error_file_id) {
        try {
          await processFailedBatch({ col, fileId: status.error_file_id });
        } catch (err) {
          logger.error({ err, batchId }, 'Failed to process failed chunks');
        }
      }

      if (status.status !== 'completed') {
        logger.warn({ batchId, status: status.status }, 'Batch not ready');
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

      logger.info({ batchId }, 'Processing completed batch');

      async function loadChunks(vectorIds) {
        const queries = vectorIds.map(vectorId => {
          const [source, rest] = vectorId.split('__p');
          const [pageStr, cid] = rest.split('__c');
          const page_counter = parseInt(pageStr, 10);
          const chunk_id = parseInt(cid, 10);
          return { source, page_counter, chunk_id };
        });
        const chunks = await col.chunks.find({ $or: queries }).toArray();
        return chunks;
      }

      const chunkEmbeddings = [];
      await pipeline(outStream, split2(JSON.parse), async function (source) {
        for await (const row of source) {
          chunkEmbeddings.push({
            vectorId: row.custom_id,
            embedding: row.response.body.data[0].embedding,
          });
        }
      });

      logger.info({ chunkEmbeddings: chunkEmbeddings.length }, 'Chunk embeddings loaded');

      const vectorIdByUUID = {};
      const chunks = await loadChunks(chunkEmbeddings.map(chunk => chunk.vectorId));
      const weaviateObjects = chunks.map((chunk, index) => {
        const id = uuidv4();
        vectorIdByUUID[id] = chunkEmbeddings[index].vectorId;
        processedSources.add(chunk.source);
        return {
          class: mapClass(chunk.metadata.data_type),
          id,
          vector: chunkEmbeddings[index].embedding,
          properties: {
            chunk_id: chunk.chunk_id,
            page_counter: chunk.page_counter,
            source: chunk.source,
            html_content: chunk.html_content,
            rag_timestamp: new Date(),
            metadata_data_type: chunk.metadata.data_type,
            metadata_timestamp: chunk.metadata.timestamp,
            metadata_crawl_depth: chunk.metadata.crawl_depth,
            metadata_title: chunk.metadata.title,
          },
        };
      });

      logger.info({ weaviateObjects: weaviateObjects.length }, 'Weaviate objects prepared');

      const doneVectorIds = [];
      for (let i = 0; i < weaviateObjects.length; i += 1000) {
        const batch = weaviateObjects.slice(i, i + 1000);
        const responses = await wv.batch.objectsBatcher().withObjects(batch).do();
        for (const response of responses) {
          if (response.result.status === 'SUCCESS') {
            doneVectorIds.push(vectorIdByUUID[response.id]);
          }
        }
        await col.embIndex.updateMany(
          { chunk_id: { $in: doneVectorIds } },
          { $set: { timestamp: new Date() } },
        );
        await col.ragReady.updateMany(
          { chunk_id: { $in: doneVectorIds } },
          { $set: { timestamp: new Date() } },
          { upsert: true },
        );
      }

      processedBatches++;
    }

    logger.warn({ processedBatches }, 'Batch processed fully');

    for (const src of processedSources) {
      const totalChunks = await col.chunks.countDocuments({ source: src });
      const doneChunks = await col.ragReady.countDocuments({
        chunk_id: { $regex: `^${src}__` },
      });
      if (totalChunks === doneChunks) {
        await col.signal.updateOne(
          { source: src, collection: mapClass(/* pick any data_type if uniform */) },
          { $setOnInsert: { timestamp: new Date() } },
          { upsert: true },
        );
        logger.info({ source: src }, 'SIGNAL_RAG doc upserted');
      }
    }

    const duration = ((Date.now() - start) / 1000).toFixed(1) + 's';
    logger.warn({ processedBatches, duration }, 'retrieveEmbed finished');
    return {
      statusCode: 200,
      body: `Processed ${processedBatches} batch(es) in ${duration}`,
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
