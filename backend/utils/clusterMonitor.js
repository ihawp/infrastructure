import { db, checkClusterHealth } from './pool.js';
import logger from './logger.js';

class ClusterMonitor {
    constructor() {
        this.isMonitoring = false;
        this.healthCheckInterval = 30000; // 30 seconds
        this.cleanupInterval = 300000; // 5 minutes
        this.intervals = [];
    }

    // Start monitoring the cluster
    start() {
        if (this.isMonitoring) {
            logger.warn('Cluster monitoring already started');
            return;
        }

        this.isMonitoring = true;
        logger.info('Starting cluster monitoring...');

        // Health check interval
        const healthInterval = setInterval(async () => {
            try {
                const health = await checkClusterHealth();
                if (!health.write || !health.read) {
                    logger.error('Cluster health check failed:', health);
                    // Implement alerting logic here
                } else {
                    logger.debug('Cluster health check passed');
                }
            } catch (error) {
                logger.error('Health check interval error:', error);
            }
        }, this.healthCheckInterval);

        // Cleanup expired tokens interval
        const cleanupInterval = setInterval(async () => {
            try {
                const result = await db.write(`
                    DELETE FROM refresh_tokens 
                    WHERE expires_at < UTC_TIMESTAMP() OR revoked = 1
                `);
                
                const magicResult = await db.write(`
                    DELETE FROM magic_keys 
                    WHERE expires_at < UTC_TIMESTAMP() OR revoked = 1
                `);

                if (result.affectedRows > 0 || magicResult.affectedRows > 0) {
                    logger.info(`Cleaned up expired tokens: ${result.affectedRows} refresh tokens, ${magicResult.affectedRows} magic keys`);
                }
            } catch (error) {
                logger.error('Token cleanup error:', error);
            }
        }, this.cleanupInterval);

        this.intervals = [healthInterval, cleanupInterval];
        logger.info('Cluster monitoring started successfully');
    }

    // Stop monitoring
    stop() {
        if (!this.isMonitoring) {
            logger.warn('Cluster monitoring not started');
            return;
        }

        this.isMonitoring = false;
        this.intervals.forEach(interval => clearInterval(interval));
        this.intervals = [];
        logger.info('Cluster monitoring stopped');
    }

    // Get cluster status
    async getStatus() {
        try {
            const health = await checkClusterHealth();
            const stats = db.getStats();
            
            return {
                monitoring: this.isMonitoring,
                health,
                stats,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            logger.error('Failed to get cluster status:', error);
            return {
                monitoring: this.isMonitoring,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    // Manual health check
    async performHealthCheck() {
        try {
            const health = await checkClusterHealth();
            logger.info('Manual health check completed:', health);
            return health;
        } catch (error) {
            logger.error('Manual health check failed:', error);
            throw error;
        }
    }

    // Test cluster connectivity
    async testConnectivity() {
        try {
            // Test read pool
            const [readResult] = await db.read('SELECT 1 as test');
            
            // Test write pool
            const [writeResult] = await db.write('SELECT 1 as test');
            
            return {
                read: readResult.length > 0,
                write: writeResult.length > 0,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            logger.error('Connectivity test failed:', error);
            throw error;
        }
    }
}

// Create singleton instance
const clusterMonitor = new ClusterMonitor();

export default clusterMonitor;
