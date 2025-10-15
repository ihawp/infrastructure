import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cookieParser from 'cookie-parser';
import apiRouter from './routers/apiRouter.js';
import cors from 'cors';
import compression from 'compression';



import { db, checkClusterHealth } from './utils/pool.js';
import clusterMonitor from './utils/clusterMonitor.js';



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

// Health check endpoint
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

// Database test endpoint
app.get('/', async (req, res) => {
    try {
        console.log('Environment check:', {
            DB_HOST: process.env.DB_HOST,
            DB_WRITE_PORT: process.env.DB_WRITE_PORT,
            DB_READ_PORT: process.env.DB_READ_PORT,
            DB_USER: process.env.DB_USER,
            DB_NAME: process.env.DB_NAME
        });

        // Test read operation (uses read pool)
        const databases = await db.read('SHOW DATABASES');
        
        // Test read operation for time
        const timeResult = await db.read('SELECT NOW()');
        
        res.status(200).json({
            success: true,
            message: 'Connected to InnoDB cluster',
            data: {
                databases: databases,
                timeResult,
                clusterInfo: {
                    readPool: 'Connected to read-only nodes',
                    writePool: 'Connected to primary node'
                },
                environment: {
                    writePort: process.env.DB_WRITE_PORT || 6446,
                    readPort: process.env.DB_READ_PORT || 6447
                }
            }
        });

    } catch (error) {
        console.error('Database test failed:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to connect to cluster',
            error: error.message,
            environment: {
                DB_HOST: process.env.DB_HOST,
                DB_WRITE_PORT: process.env.DB_WRITE_PORT,
                DB_READ_PORT: process.env.DB_READ_PORT,
                DB_USER: process.env.DB_USER,
                DB_NAME: process.env.DB_NAME
            }
        });
    }
});

/*
app.get('/', async (req, res) => {
    try {
        console.log('Adding test job...');

        const job = await bull.add(
            'test',
            { message: 'Hello World' },
            {
                removeOnComplete: true,
                removeOnFail: true,
                attempts: 3,
                backoff: {
                    type: 'exponential',
                    delay: 1000
                },
                delay: 0,
                timeout: 5000,
                jobId: 'test-' + Date.now()
            }
        );

        console.log('Job added successfully:', job.id);

        res.send('Hello World - Job added: ' + job.id);
    } catch (error) {
        console.error('Error adding job:', error);
        res.status(500).send('Error adding job: ' + error.message);
    }
});
*/

// Start cluster monitoring
clusterMonitor.start();

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\nShutting down gracefully...');
    clusterMonitor.stop();
    await db.close();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\nShutting down gracefully...');
    clusterMonitor.stop();
    await db.close();
    process.exit(0);
});

app.listen(process.env.PORT, () => {
    console.log(`Server running on port ${process.env.PORT}`);
    console.log('InnoDB Cluster monitoring started');
});