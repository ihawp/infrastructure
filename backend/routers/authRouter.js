import express from 'express';
import authControllers from '../controllers/authControllers.js';
import expressValidator from '../middleware/expressValidator.js';
import authValidator from '../middleware/authValidator.js';
import checkCredentials from '../middleware/checkCredentials.js';

const { registerController, loginController, magicController, verifyController, deleteController } =
    authControllers;
const { validateErrors } = expressValidator;
const { validatorRegister, validatorLogin, validatorMagic } = authValidator;

const authRouter = express.Router();

authRouter.post('/register', validatorRegister, validateErrors, registerController);

authRouter.post('/login', validatorLogin, validateErrors, checkCredentials, loginController);

authRouter.get('/magic', validatorMagic, validateErrors, magicController);

/* Only meant for frontend apps to call */
authRouter.post('/verify', verifyController);

authRouter.post('/delete', deleteController);

export default authRouter;
