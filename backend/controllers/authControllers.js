import authModels from '../models/authModels.js';

const authControllers = {
    registerController: async (req, res) => {
        const response = await authModels.registerModel(req, res);
        console.log(response);
        return res.status(200).json({ response });
    },
    loginController: (req, res) => {
        res.send('hello world');
    },
    magicLinkController: (req, res) => {
        res.send('hello world');
    },
    verifyTokenController: (req, res) => {
        res.send('hello world');
    }
};

export default authControllers;
