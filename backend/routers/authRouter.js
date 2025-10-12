import express from 'express';
import authControllers from '../controllers/authControllers.js';
import expressValidator from '../middleware/expressValidator.js';
import authValidator from '../middleware/authValidator.js';
const { registerController, loginController, magicController, verifyController, deleteController } = authControllers;
const { validateErrors } = expressValidator;
const { validatorRegister, validatorLogin } = authValidator;

const authRouter = express.Router();

authRouter.post('/register', validatorRegister, validateErrors, registerController);

authRouter.post('/login', validatorLogin, validateErrors, loginController);

authRouter.post('/magic', magicController);

authRouter.post('/verify', verifyController);

authRouter.post('/delete', deleteController);

export default authRouter;