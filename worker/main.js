import dotenv from 'dotenv';
dotenv.config();

import Bull from 'bull';
import logger from './logger.js';

async function register(user) {
    console.log('Registering user', user);
    logger.info({ verificationEmail: 'sent', user });
    console.log('User registered');
    return true;
}
  
const host = process.env.REDIS_HOST;
const port = process.env.REDIS_PORT;
const password = process.env.REDIS_PASSWORD;

const authQueue = new Bull('auth', { redis: { host, port, password } });
authQueue.process(async (job) => {
  if (job.data.type === 'register') await register(job.data.user);
});

console.log('Worker started');
