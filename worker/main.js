import dotenv from 'dotenv';
dotenv.config();

import { Worker } from 'bullmq';
import Redis from 'ioredis';
import logger from './logger.js';
import register from './functions/register.js';
import magicKey from './functions/magicKey.js';

  // Wrapper for ioredis
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

const worker = new Worker(
    '{cluster}', // queue name
    async (job) => {
        console.log('Processing job:', job.name, job.id);
        // process jobs based on name
        switch (job.name) {
            case 'sendMagicLink':
                return magicKey.send(job);
                break;
            case 'registerUser':
                return register.registerUser(job);
                break;
            case 'test':
                console.log('test job received');
                console.log(job.data);
                return 'done';
                break;
            default:
                throw new Error(`Unknown job name: ${job.name}`);
                break;
        }
    }, { connection: cluster }
);

worker.on('completed', (job, returnvalue) => {
  console.log(`Job ${job.id} completed`, returnvalue);
});

worker.on('failed', (job, err) => {
  console.log(`Job ${job.id} failed`, err);
});

worker.on('error', (err) => console.log('Worker internal error:', err));

worker.on('closed', () => console.log('Worker closed'));

worker.on('listening', () => console.log('Worker listening'));

worker.on('restarting', () => console.log('Worker restarting'));

worker.on('restarted', () => console.log('Worker restarted'));