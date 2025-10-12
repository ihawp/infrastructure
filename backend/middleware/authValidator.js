import { body, param } from 'express-validator';

/* Delete and Verify do not need validation. */
const authValidator = {
    validatorRegister: [
        body('name').notEmpty().withMessage('Name is required'),
        body('firstname').notEmpty().withMessage('Firstname is required'),
        body('lastname').notEmpty().withMessage('Lastname is required')
    ],
    validatorLogin: [
        body('email').notEmpty().withMessage('Email is required'),
        body('password').notEmpty().withMessage('Password is required')
    ],
    validatorMagic: [
        param('key').isLength({ min: 64, max: 64 }).notEmpty().withMessage('Key is required'),
        param('id').notEmpty().withMessage('ID is required')
    ]
};

export default authValidator;
