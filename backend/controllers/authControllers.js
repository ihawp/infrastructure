import authModels from '../models/authModels.js';
import queue from '../utils/bull.js';
import logger from '../utils/logger.js';

const {} = authModels;

const authControllers = {
    registerController: async (req, res) => {

        const job = await queue.add('registerUser', {
            name: req.body.name,
            firstname: req.body.firstname,
            lastname: req.body.lastname
        }, {
            removeOnComplete: true,
            removeOnFail: true,
            jobId: req.body.id 
        });
        
        return res
            .status(200)
            .json({
                success: true,
                redirect: 'check-email',
                data: {
                    jobId: job.id
                }
            });
    },

    loginController: async (req, res) => {

        // *** REQ.BODY.ID IS NOT YET DEFINED AT THIS POINT ***

        try {

            /*
            await queue.add('processPayment', { userId, amount }, {
                removeOnComplete: true,   // completed jobs can be cleaned
                attempts: 5,              // retry automatically a few times
                backoff: 10000            // 10s delay between retries
            });
            */

            const job = await bull.add('sendMagicLink', {
                email: req.body.email,
                id: req.body.id
            }, {
                removeOnComplete: true,
                removeOnFail: true
            });
    
            return res.status(200).json({
                success: true,
                redirect: 'check-email',
                data: {
                    jobId: job.id
                }
            });

        } catch (error) {

            logger.error(error);
        
            return res.status(500).json({
                success: false,
                message: 'Internal server error'
            });
        
        }
    },

    magicController: (req, res) => {
        return res
            .status(200)
            .json({
                success: true,
                redirect: 'home'
            });
    },

    verifyController: (req, res) => {
        return res
            .status(200)
            .json({
                success: true,
                id: req.body.id
            });
    },

    deleteController: (req, res) => {
        return res
            .status(200)
            .json({
                success: true,
                redirect: 'check-email'
            });
    }
};

export default authControllers;
