import dotenv from 'dotenv';
dotenv.config();

import Bull from 'bull';
import logger from './logger.js';

const host = process.env.REDIS_HOST;
const port = process.env.REDIS_PORT;
const password = process.env.REDIS_PASSWORD;

const bull = new Bull('auth', {
    redis: {
        host,
        port,
        password
    }
});

bull.process(async (job) => {
    logger.info(`Processing job ${job.id}`);
});

bull.on('completed', (job) => {
    logger.info({ job: job.id, status: 'completed' });
});

bull.on('failed', (job) => {

});

bull.on('waiting', (jobId) => {

});

bull.on('active', (job) => {

});

bull.on('stalled', (job) => {

});

bull.on('removed', (job) => {

});

bull.on('cleaned', (jobs) => {

});

bull.on('paused', () => {

});

bull.on('resumed', () => {

});

bull.on('drained', () => {

});

export default bull;