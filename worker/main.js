import dotenv from 'dotenv';
dotenv.config();

import { Worker } from 'bullmq';
import Redis from 'ioredis';
import { redisLogger } from './logger.js';
import register from './functions/register.js';
import magicKey from './functions/magicKey.js';

const cluster = new Redis.Cluster([
  { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT, password: process.env.REDIS_PASSWORD },
  { host: process.env.REDIS_HOST2, port: process.env.REDIS_PORT2, password: process.env.REDIS_PASSWORD2 },
  { host: process.env.REDIS_HOST3, port: process.env.REDIS_PORT3, password: process.env.REDIS_PASSWORD3 },
  { host: process.env.REDIS_HOST4, port: process.env.REDIS_PORT4, password: process.env.REDIS_PASSWORD4 },
  { host: process.env.REDIS_HOST5, port: process.env.REDIS_PORT5, password: process.env.REDIS_PASSWORD5 },
  { host: process.env.REDIS_HOST6, port: process.env.REDIS_PORT6, password: process.env.REDIS_PASSWORD6 },
], {
  clusterRetryStrategy: times => Math.min(100 + times * 2, 2000), // optional retry strategy
  scaleReads: 'slave',
  enableReadyCheck: false,
  logger: redisLogger
});

// Create multiple workers, each worker uses it's own thread
// and connection to Redis client.

const worker = new Worker('Cluster',
    async (job) => {

        console.log(job.name);
      
        switch (job.name) {
            case 'sendMagicLink':
                return await magicKey.send(job);
            case 'registerUser':
                return await register.registerUser(job);
            case 'test':
                console.log('test job received');
                return 'done';
            default:
                logger.error(`Unknown job name: ${job.name}`);
                throw new Error(`Unknown job name: ${job.name}`);
        }
    }, { connection: cluster, prefix: '{cluster}' }
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