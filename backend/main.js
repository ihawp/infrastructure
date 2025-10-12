import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';
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

app.get('/', async (req, res) => {
    try {
        console.log('Adding test job...');
        
        const job = await bull.add('test', { message: 'Hello World' });
        
        console.log('Job added successfully:', job.id);
        
        res.send('Hello World - Job added: ' + job.id);
    } catch (error) {
        console.error('Error adding job:', error);
        res.status(500).send('Error adding job: ' + error.message);
    }
});

app.listen(process.env.PORT);