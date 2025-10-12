import authModels from '../models/authModels.js';
const {} = authModels;

const authControllers = {
    registerController: async (req, res) => {
        return res
            .status(200)
            .json({
                success: true,
                redirect: 'check-email'
            });
    },

    loginController: (req, res) => {
        return res
            .status(200)
            .json({
                success: true,
                redirect: 'check-email'
            });
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
