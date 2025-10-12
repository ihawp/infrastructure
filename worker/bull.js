import dotenv from 'dotenv';
dotenv.config();

import { Worker } from 'bullmq';
import Redis from 'ioredis';
import logger from './logger.js';
import register from './functions/register.js';
import magicKey from './functions/magicKey.js';

const cluster = new Redis.Cluster([
  { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT, password: process.env.REDIS_PASSWORD },
  { host: process.env.REDIS_HOST2, port: process.env.REDIS_PORT2, password: process.env.REDIS_PASSWORD2 },
  { host: process.env.REDIS_HOST3, port: process.env.REDIS_PORT3, password: process.env.REDIS_PASSWORD3 },
]);

const worker = new Worker(
  'auth', // queue name
  async (job) => {
    // process jobs based on name
    switch (job.name) {
      case 'sendMagicLink':
        return magicKey.send(job);
      case 'registerUser':
        return register.registerUser(job);
      case 'test':
        console.log('test job received');
        console.log(job.data);
        return 'done';
      default:
        throw new Error(`Unknown job name: ${job.name}`);
    }
  },
  { connection: cluster }
);

worker.on('completed', (job, returnvalue) => {
  console.log(`Job ${job.id} completed`, returnvalue);
});

worker.on('failed', (job, err) => {
  console.error(`Job ${job.id} failed`, err);
});

export default worker;