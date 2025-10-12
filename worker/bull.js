import dotenv from 'dotenv';
dotenv.config();

import Bull from 'bull';
import Redis from 'ioredis';
import logger from './logger.js';
import register from './functions/register.js';
import magicKey from './functions/magicKey.js';
  
const cluster = new Redis.Cluster([
    { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT, password: process.env.REDIS_PASSWORD },
    { host: process.env.REDIS_HOST2, port: process.env.REDIS_PORT2, password: process.env.REDIS_PASSWORD2 },
    { host: process.env.REDIS_HOST3, port: process.env.REDIS_PORT3, password: process.env.REDIS_PASSWORD3 },
]);

const bull = new Bull('auth', () => { createClient: cluster });

bull.process('sendMagicLink', async (job) => magicKey.send(job));

bull.process('registerUser', async (job) => register.registerUser(job));

const test = {
    test: async (job) => {
        console.log(job.data);
    }
}

// the name is the ideal identifier for the job, duh
// and then the data is the data that is passed to the job
// and that data is useful for the job inherently, because
// the data that is passed is exactly what is expected for
// the job.

// lock is per job id, lockDuration, removeOnComplete
bull.process('test', async (job) => test.test(job));

export default bull;