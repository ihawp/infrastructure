import dotenv from 'dotenv';
dotenv.config();

import mysql from 'mysql2/promise';
import logger from './logger.js';

// InnoDB Cluster Configuration

// Read/Write Pool (Primary Node)
const writePool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_WRITE_PORT || 6446,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 15,
    maxIdle: 10,
    idleTimeout: 60000,
    queueLimit: 0,
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
    multipleStatements: false,
    supportBigNumbers: true,
    bigNumberStrings: true
});

// Read-Only Pool (Secondary Node)
// This will be moved to the worker process
const readPool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_READ_PORT || 6447,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 20,
    maxIdle: 15,
    idleTimeout: 60000,
    queueLimit: 0,
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
    multipleStatements: false,
    supportBigNumbers: true,
    bigNumberStrings: true
});

const db = {

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
export { db, writePool, readPool };
