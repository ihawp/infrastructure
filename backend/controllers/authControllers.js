import authModels from '../models/authModels.js';
import queue from '../utils/bull.js';
import logger from '../utils/logger.js';

const {} = authModels;

const authControllers = {
    registerController: async (req, res) => {

        // add the user to the database

        // create job to send email to user with magic login link (pass user id, email)
            // inside job:
                // send magic key to redis like magickey:${userId} with 10 MINUTE EXPIRY.
                // send email to user

        // create job to add user to DBX.
        



        try {
            const job = await bull.add(
                'registerUser',
                {
                    // both required for the email, in the function dealing with the job
                    // a token will be generated for the user, hashed, and saved to redis, this
                    // token will be sent to the user, the token will be set to EXPIRE after 10
                    // minutes.
                    email: req.body.email,
                    id: req.body.id
                },
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
                    jobId: `${req.body.id}-` + Date.now()
                }
            );

            res.status(200).json({
                success: true,
                data: {
                    jobId: job.id
                }
            });
        } catch (error) {
            console.error('Error adding job:', error);
            res.status(500).send('Error adding job: ' + error.message);
        }
    },

    loginController: async (req, res) => {
        
        const { id, username, email } = req.user;

        // create the job
        // and then the worker will create the magic token
        // for the user and add it to redis.

        // when the user clicks the magic link
        // from their email, it will take them to the
        // /api/auth/magic route, where token and id are
        // passed as parms.

        const job = await bull.add(
            'loginUser',
            {
                id,
                username,
                email
            }
        );

        if (!job) {
            return res
                .status(500)
                .json({
                    success: false,
                    message: 'Error adding job'
                });
        }

        return res
            .status(200)
            .json({
                success: true,
                data: {
                    jobId: job.id
                }
            })
    },

    magicController: (req, res) => {

        const { token, id } = req.params;

        


        return res.status(200).json({
            success: true,
            redirect: 'home'
        });
    },

    verifyController: (req, res) => {
        return res.status(200).json({
            success: true,
            id: req.body.id
        });
    },

    deleteController: (req, res) => {
        return res.status(200).json({
            success: true,
            redirect: 'check-email'
        });
    }
};

export default authControllers;
