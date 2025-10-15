import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';
import cors from 'cors';
import compression from 'compression';

import { db } from './utils/pool.js';

const app = express();

app.use(compression());
app.use(cors({
    origin: process.env.FRONTEND_URL,
    credentials: true,
    origin: '*'
}));
app.use(cookieParser());
app.use(express.json());

// Routes
app.use('/api', apiRouter);

app.get('/health', async (req, res) => {
    try {
        const health = await checkClusterHealth();
        const stats = db.getStats();
        
        res.status(200).json({
            success: true,
            message: 'Cluster health check',
            data: {
                cluster: health,
                poolStats: stats,
                timestamp: new Date().toISOString()
            }
        });
    } catch (error) {
        console.error('Health check failed:', error);
        res.status(500).json({
            success: false,
            message: 'Health check failed',
            error: error.message
        });
    }
});

const graceful = async () => {
    await db.close();
    process.exit(0);
}

process.on('SIGINT', graceful);
process.on('SIGTERM', graceful);

app.listen(process.env.PORT);