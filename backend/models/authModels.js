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
    login: async (email) => {
        try {
            const [response] = await pool.query(`
                SELECT id, username, password
                FROM accounts 
                WHERE email = ?`,
                [email]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return error;
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

    insertMagicKey: async (key, id) => {
        try {
            const [response] = await pool.query(`
                INSERT INTO magic_keys
                (user_id, token_hash, expires_at)
                VALUES (?, ?, UTC_TIMESTAMP() + INTERVAL 10 MINUTE)`,
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
                SELECT id, user_id FROM magic_keys
                WHERE magic_link_key = ? AND expires_at > UTC_TIMESTAMP()`,
                [key]
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
                DELETE FROM magic_keys
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
                SET revoked = 1, expires = UTC_TIMESTAMP()
                WHERE id = ? LIMIT 1`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error(error);
            return false;
        }
    },


};

export default authModels;
