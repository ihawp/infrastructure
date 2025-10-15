import dotenv from 'dotenv';
dotenv.config();

import { Queue } from 'bullmq';
import Redis from 'ioredis';
import { redisLogger } from './logger.js';

const cluster = new Redis.Cluster(
    [
        {
            host: process.env.REDIS_HOST,
            port: process.env.REDIS_PORT,
            password: process.env.REDIS_PASSWORD
        },
        {
            host: process.env.REDIS_HOST2,
            port: process.env.REDIS_PORT2,
            password: process.env.REDIS_PASSWORD2
        },
        {
            host: process.env.REDIS_HOST3,
            port: process.env.REDIS_PORT3,
            password: process.env.REDIS_PASSWORD3
        },
        {
            host: process.env.REDIS_HOST4,
            port: process.env.REDIS_PORT4,
            password: process.env.REDIS_PASSWORD4
        },
        {
            host: process.env.REDIS_HOST5,
            port: process.env.REDIS_PORT5,
            password: process.env.REDIS_PASSWORD5
        },
        {
            host: process.env.REDIS_HOST6,
            port: process.env.REDIS_PORT6,
            password: process.env.REDIS_PASSWORD6
        }
    ],
    {
        clusterRetryStrategy: (times) => Math.min(100 + times * 2, 2000), // optional retry strategy
        scaleReads: 'slave',
        enableReadyCheck: false,
        logger: redisLogger
    }
);

const queue = new Queue('{cluster}', { connection: cluster });

export default queue;
