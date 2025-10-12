import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';

import bull from './utils/bull.js';

const app = express();

app.use(cookieParser());
app.use(express.json());

// Main gateway
app.use('/api', apiRouter);

app.get('/', (req, res) => {
    bull.add({ type: "register",verified: 0, user: { id: 1, email: 'test@test.com' } });
    res.send('Hello World');
});

app.listen(process.env.PORT);
