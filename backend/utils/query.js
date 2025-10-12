import pool from './pool.js';
import logger from './logger.js';

const transaction = async (query, values) => {
    const connection = await pool.getConnection();
    await connection.beginTransaction();
    try {
        const [response] = await connection.query(query, values);
        await connection.commit();
        logger.info('Query executed successfully');
        return response;
    } catch (error) {
        logger.error(error);
        logger.error('Query execution failed');
        await connection.rollback();
        throw error;
    } finally {
        logger.info('Connection released');
        connection.release();
    }
}

export default transaction;