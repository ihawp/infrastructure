import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';

const app = express();

app.use(cookieParser());
app.use(express.json());

// Main gateway
app.use('/api', apiRouter);

app.get('/', (req, res) => {
    res.send('Hello World');
});

app.listen(process.env.PORT);
