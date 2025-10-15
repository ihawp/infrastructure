import dotenv from 'dotenv';
dotenv.config();

import mysql from 'mysql2/promise';
import logger from './logger.js';

// InnoDB Cluster Configuration
// Using MySQL Router ports for cluster-aware connections

// Read/Write Pool (Primary Node)
const writePool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_WRITE_PORT || 6446, // MySQL Router read/write port
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 15, // Higher limit for write operations
    maxIdle: 10,
    idleTimeout: 60000,
    queueLimit: 0,
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
    multipleStatements: false,
    supportBigNumbers: true,
    bigNumberStrings: true
});

// Read-Only Pool (Secondary Nodes)
const readPool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_READ_PORT || 6447, // MySQL Router read-only port
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 20, // Higher limit for read operations
    maxIdle: 15,
    idleTimeout: 60000,
    queueLimit: 0,
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
    multipleStatements: false,
    supportBigNumbers: true,
    bigNumberStrings: true
});

// CLuster Health Check
const checkClusterHealth = async () => {
    try {
        // Check write pool (primary)
        const [writeResult] = await writePool.query('SELECT 1 as health_check');
        logger.info('Write pool health check passed');
        
        // Check read pool (secondary)
        const [readResult] = await readPool.query('SELECT 1 as health_check');
        logger.info('Read pool health check passed');
        
        return { write: true, read: true };
    } catch (error) {
        logger.error('Cluster health check failed:', error);
        return { write: false, read: false, error: error.message };
    }
};

// Database OPerations with Cluster Awareness
const db = {
    // Write operations (INSERT, UPDATE, DELETE)
    write: async (query, params = []) => {
        try {
            const [result] = await writePool.execute(query, params);
            logger.debug('Write operation executed:', { query: query.substring(0, 100), affectedRows: result.affectedRows });
            return result;
        } catch (error) {
            logger.error('Write operation failed:', { query, params, error: error.message });
            throw error;
        }
    },

    // Read operations (SELECT)
    read: async (query, params = []) => {
        try {
            const [result] = await readPool.execute(query, params);
            logger.debug('Read operation executed:', { query: query.substring(0, 100), rows: result.length });
            return result;
        } catch (error) {
            logger.error('Read operation failed:', { query, params, error: error.message });
            throw error;
        }
    },

    // Transaction operations (always use write pool)
    transaction: async (callback) => {
        const connection = await writePool.getConnection();
        try {
            await connection.beginTransaction();
            const result = await callback(connection);
            await connection.commit();
            return result;
        } catch (error) {
            await connection.rollback();
            throw error;
        } finally {
            connection.release();
        }
    },

    // Health check
    health: checkClusterHealth,

    // Get pool statistics
    getStats: () => ({
        write: {
            totalConnections: writePool.pool._allConnections.length,
            freeConnections: writePool.pool._freeConnections.length,
            acquiringConnections: writePool.pool._acquiringConnections.length
        },
        read: {
            totalConnections: readPool.pool._allConnections.length,
            freeConnections: readPool.pool._freeConnections.length,
            acquiringConnections: readPool.pool._acquiringConnections.length
        }
    }),

    // Graceful shutdown
    close: async () => {
        await Promise.all([
            writePool.end(),
            readPool.end()
        ]);
        logger.info('Database pools closed');
    }
};

// Legacy support - default to write pool for backward compatibility
const pool = writePool;

export default pool;
export { db, writePool, readPool, checkClusterHealth };
