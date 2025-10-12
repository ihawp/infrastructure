import pool from '../utils/pool.js';
import logger from '../utils/logger.js';

const authModels = {
    register: async (x) => {
        try {
            const [response] = await pool.query(`
                INSERT INTO accounts 
                (x) 
                VALUES (?)`,
                [x]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return error;
        }
    },
    login: async (x) => {
        try {
            const [response] = await pool.query(`
                INSERT INTO accounts 
                (x) 
                VALUES (?)`,
                [x]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return error;
        }
    },
    insertMagicKey: async (key, id) => {
        try {
            const [response] = await pool.query(`
                UPDATE accounts
                SET magic_link_key = ?
                WHERE id = ?`,
                [key, id]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    },
    checkMagicKey: async (key) => {
        try {
            const [response] = await pool.query(`
                SELECT id FROM accounts
                WHERE magic_link_key = ?`,
                [key]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    },
    deleteAccount: async (id) => {
        try {
            const [response] = await pool.query(`
                DELETE FROM accounts
                WHERE id = ?`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    },
    deleteMagicKey: async (id) => {
        try {
            const [response] = await pool.query(`
                UPDATE accounts
                SET magic_link_key = NULL
                WHERE id = ?`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    },
    disableRefreshToken: async (id) => {
        try {
            const [response] = await pool.query(`
                UPDATE refresh_tokens
                SET revoked = 1, expires = CURRENT_TIMESTAMP
                WHERE id = ? LIMIT 1`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    }
};

export default authModels;
