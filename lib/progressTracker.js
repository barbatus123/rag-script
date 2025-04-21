import { logger } from './logger.js';

export class ProgressTracker {
  constructor(processId) {
    this.processId = processId;
    this.startTime = Date.now();
    this.metrics = {
      totalChunks: 0,
      processedChunks: 0,
      failedChunks: 0,
      skippedChunks: 0,
      totalBatches: 0,
      completedBatches: 0,
      failedBatches: 0,
      totalTokens: 0,
      averageTokensPerChunk: 0,
    };
  }

  incrementMetric(metric, value = 1) {
    this.metrics[metric] += value;
    this.logProgress();
  }

  updateTokens(tokens) {
    this.metrics.totalTokens += tokens;
    this.metrics.averageTokensPerChunk = this.metrics.totalTokens / this.metrics.processedChunks;
  }

  logProgress() {
    const elapsed = (Date.now() - this.startTime) / 1000;
    const progress =
      this.metrics.totalChunks > 0
        ? ((this.metrics.processedChunks / this.metrics.totalChunks) * 100).toFixed(2)
        : 0;

    logger.info(
      {
        processId: this.processId,
        progress: `${progress}%`,
        elapsed: `${elapsed.toFixed(1)}s`,
        metrics: this.metrics,
        rate:
          this.metrics.processedChunks > 0
            ? `${(this.metrics.processedChunks / elapsed).toFixed(2)} chunks/s`
            : '0 chunks/s',
      },
      'Progress update',
    );
  }

  getMetrics() {
    return {
      ...this.metrics,
      elapsedSeconds: (Date.now() - this.startTime) / 1000,
    };
  }
}
