import dotenv from 'dotenv';
dotenv.config();

import { Queue } from 'bullmq';
import Redis from 'ioredis';
import logger from './logger.js';

const redisLogger = {
    debug: (msg) => logger.debug(msg),
    info: (msg) => logger.info(msg),
    warn: (msg) => logger.warn(msg),
    error: (msg) => logger.error(msg)
};
  
const cluster = new Redis.Cluster([
    { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT, password: process.env.REDIS_PASSWORD },
    { host: process.env.REDIS_HOST2, port: process.env.REDIS_PORT2, password: process.env.REDIS_PASSWORD2 },
    { host: process.env.REDIS_HOST3, port: process.env.REDIS_PORT3, password: process.env.REDIS_PASSWORD3 },
], {
    clusterRetryStrategy: times => Math.min(100 + times * 2, 2000), // optional retry strategy
    scaleReads: 'master', // change to slave when available
    enableReadyCheck: false,
    logger: redisLogger
});

const queue = new Queue('{cluster}', { connection: cluster });

export default queue;