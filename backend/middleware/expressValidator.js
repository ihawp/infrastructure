import { validationResult } from 'express-validator';

const expressValidator = {
    validateErrors: (req, res, next) => {
        const errors = validationResult(req);
        if (errors.isEmpty()) {
            return next();
        }
        return res.status(400).json({ errors: errors.array() });
    }
};

export default expressValidator;
