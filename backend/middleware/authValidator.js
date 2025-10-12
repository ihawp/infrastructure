import { body } from 'express-validator';

const authValidator = {
    validatorRegister: [
        body('name').notEmpty().withMessage('Name is required'),
        body('firstname').notEmpty().withMessage('Firstname is required'),
        body('lastname').notEmpty().withMessage('Lastname is required')
    ],
    validatorLogin: [
        body('email').notEmpty().withMessage('Email is required'),
        body('password').notEmpty().withMessage('Password is required')
    ]
};

export default authValidator;
