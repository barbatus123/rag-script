import pino from 'pino';
import pretty from 'pino-pretty';

const stream = process.env.NODE_ENV === 'development'
    ? pretty({ colorize: true })
    : process.stdout; // DigitalOcean Functions forward stdout to the logging UI

export const logger = pino(
    {
        level: process.env.LOG_LEVEL || 'info',
        base: { service: 'rag-embedding-pipeline' },
        timestamp: pino.stdTimeFunctions.isoTime
    },
    stream
);