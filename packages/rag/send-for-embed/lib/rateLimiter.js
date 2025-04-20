import { logger } from './logger.js';

export class RateLimiter {
    constructor(requestsPerMinute) {
        this.requestsPerMinute = requestsPerMinute;
        this.requests = [];
        this.lastCleanup = Date.now();
    }

    async waitForSlot() {
        const now = Date.now();
        
        // Clean up old requests (older than 1 minute)
        if (now - this.lastCleanup > 60000) {
            this.requests = this.requests.filter(time => now - time < 60000);
            this.lastCleanup = now;
        }

        // If we're under the limit, proceed immediately
        if (this.requests.length < this.requestsPerMinute) {
            this.requests.push(now);
            return;
        }

        // Calculate wait time until the oldest request expires
        const oldestRequest = this.requests[0];
        const waitTime = 60000 - (now - oldestRequest);
        
        if (waitTime > 0) {
            logger.warn({ waitTime }, 'Rate limit reached, waiting');
            await new Promise(resolve => setTimeout(resolve, waitTime));
            this.requests.shift(); // Remove the oldest request
            this.requests.push(Date.now());
        }
    }

    getRemainingRequests() {
        return Math.max(0, this.requestsPerMinute - this.requests.length);
    }
} 