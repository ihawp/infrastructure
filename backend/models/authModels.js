import { db } from '../utils/pool.js';
import logger from '../utils/logger.js';

const authModels = {
    // User registration - WRITE operation
    register: async (userData) => {
        try {
            const response = await db.write(
                `INSERT INTO accounts 
                (role, username, pfp_url, password_hash, email, email_verified, email_verified_at, account_created) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    userData.role || 'user',
                    userData.username,
                    userData.pfp_url || null,
                    userData.password_hash,
                    userData.email,
                    userData.email_verified || false,
                    userData.email_verified_at || null,
                    userData.account_created || new Date()
                ]
            );
            return response;
        } catch (error) {
            logger.error('Registration failed:', error);
            return error;
        }
    },

    // User login - READ operation
    login: async (email) => {
        try {
            const response = await db.read(
                `SELECT id, role, username, pfp_url, password_hash, email, email_verified, email_verified_at, account_created
                FROM accounts 
                WHERE email = ?`,
                [email]
            );
            return response;
        } catch (error) {
            logger.error('Login query failed:', error);
            return error;
        }
    },

    // Get user by ID - READ operation
    getUserById: async (id) => {
        try {
            const response = await db.read(
                `SELECT id, role, username, pfp_url, email, email_verified, email_verified_at, account_created
                FROM accounts 
                WHERE id = ?`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error('Get user by ID failed:', error);
            return error;
        }
    },

    // Update user - WRITE operation
    updateUser: async (id, updateData) => {
        try {
            const fields = [];
            const values = [];
            
            Object.keys(updateData).forEach(key => {
                if (updateData[key] !== undefined) {
                    fields.push(`${key} = ?`);
                    values.push(updateData[key]);
                }
            });
            
            if (fields.length === 0) {
                throw new Error('No fields to update');
            }
            
            values.push(id);
            
            const response = await db.write(
                `UPDATE accounts 
                SET ${fields.join(', ')}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?`,
                values
            );
            return response;
        } catch (error) {
            logger.error('Update user failed:', error);
            return error;
        }
    },

    // Delete account - WRITE operation
    deleteAccount: async (id) => {
        try {
            const response = await db.write(
                `DELETE FROM accounts WHERE id = ?`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error('Delete account failed:', error);
            return false;
        }
    },

    // Refresh Token Operations - WRITE operations
    createRefreshToken: async (userId, tokenHash, expiresAt) => {
        try {
            const response = await db.write(
                `INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (?, ?, ?)`,
                [userId, tokenHash, expiresAt]
            );
            return response;
        } catch (error) {
            logger.error('Create refresh token failed:', error);
            return error;
        }
    },

    getRefreshToken: async (tokenHash) => {
        try {
            const response = await db.read(
                `SELECT id, user_id, token_hash, created_at, expires_at, used_at, revoked
                FROM refresh_tokens 
                WHERE token_hash = ? AND revoked = 0 AND expires_at > UTC_TIMESTAMP()`,
                [tokenHash]
            );
            return response;
        } catch (error) {
            logger.error('Get refresh token failed:', error);
            return error;
        }
    },

    updateRefreshToken: async (id, updateData) => {
        try {
            const fields = [];
            const values = [];
            
            Object.keys(updateData).forEach(key => {
                if (updateData[key] !== undefined) {
                    fields.push(`${key} = ?`);
                    values.push(updateData[key]);
                }
            });
            
            if (fields.length === 0) {
                throw new Error('No fields to update');
            }
            
            values.push(id);
            
            const response = await db.write(
                `UPDATE refresh_tokens 
                SET ${fields.join(', ')}
                WHERE id = ?`,
                values
            );
            return response;
        } catch (error) {
            logger.error('Update refresh token failed:', error);
            return error;
        }
    },

    disableRefreshToken: async (id) => {
        try {
            const response = await db.write(
                `UPDATE refresh_tokens
                SET revoked = 1, used_at = UTC_TIMESTAMP()
                WHERE id = ? LIMIT 1`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error('Disable refresh token failed:', error);
            return false;
        }
    },

    // Magic Key Operations - WRITE operations
    createMagicKey: async (userId, tokenHash, expiresAt) => {
        try {
            const response = await db.write(
                `INSERT INTO magic_keys (user_id, token_hash, expires_at)
                VALUES (?, ?, ?)`,
                [userId, tokenHash, expiresAt]
            );
            return response;
        } catch (error) {
            logger.error('Create magic key failed:', error);
            return error;
        }
    },

    getMagicKey: async (tokenHash) => {
        try {
            const response = await db.read(
                `SELECT id, user_id, token_hash, created_at, expires_at, used_at, revoked
                FROM magic_keys 
                WHERE token_hash = ? AND revoked = 0 AND expires_at > UTC_TIMESTAMP()`,
                [tokenHash]
            );
            return response;
        } catch (error) {
            logger.error('Get magic key failed:', error);
            return error;
        }
    },

    useMagicKey: async (id) => {
        try {
            const response = await db.write(
                `UPDATE magic_keys
                SET used_at = UTC_TIMESTAMP(), revoked = 1
                WHERE id = ? LIMIT 1`,
                [id]
            );
            return response;
        } catch (error) {
            logger.error('Use magic key failed:', error);
            return false;
        }
    },

    // Cleanup expired tokens - WRITE operations
    cleanupExpiredTokens: async () => {
        try {
            const refreshResult = await db.write(
                `DELETE FROM refresh_tokens 
                WHERE expires_at < UTC_TIMESTAMP() OR revoked = 1`
            );
            
            const magicResult = await db.write(
                `DELETE FROM magic_keys 
                WHERE expires_at < UTC_TIMESTAMP() OR revoked = 1`
            );
            
            return {
                refreshTokensDeleted: refreshResult.affectedRows,
                magicKeysDeleted: magicResult.affectedRows
            };
        } catch (error) {
            logger.error('Cleanup expired tokens failed:', error);
            return false;
        }
    }
};

export default authModels;
