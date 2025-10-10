import express from 'express';
import authControllers from '../controllers/authControllers.js';
import expressValidator from '../middleware/expressValidator.js';
import authValidator from '../middleware/authValidator.js';
const { registerController, loginController } = authControllers;
const { validateErrors } = expressValidator;
const { validatorRegister, validatorLogin } = authValidator;

const authRouter = express.Router();

authRouter.post('/register', validatorRegister, validateErrors, registerController);

authRouter.post('/login', validatorLogin, validateErrors, loginController);

export default authRouter;
