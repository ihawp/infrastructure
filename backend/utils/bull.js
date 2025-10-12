import dotenv from 'dotenv';
dotenv.config();

import Bull from 'bull';
import Redis from 'ioredis';
  
const cluster = new Redis.Cluster([
    { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT, password: process.env.REDIS_PASSWORD },
    { host: process.env.REDIS_HOST2, port: process.env.REDIS_PORT2, password: process.env.REDIS_PASSWORD2 },
    { host: process.env.REDIS_HOST3, port: process.env.REDIS_PORT3, password: process.env.REDIS_PASSWORD3 },
]);

const bull = new Bull('auth', { createClient: () => cluster });

export default bull;