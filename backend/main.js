import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';
import logger from './utils/logger.js';
import cors from 'cors';
import bull from './utils/bull.js';

const app = express();

app.use(cors({
    origin: process.env.FRONTEND_URL,
    credentials: true
}));
app.use(cookieParser());
app.use(express.json());

// Main gateway
app.use('/api', apiRouter);

app.get('/', (req, res) => {

    console.log('adding job');

    bull.add('test', { message: 'Hello World' });

    console.log('job added?');

    res.send('Hello World');
});

app.listen(process.env.PORT);