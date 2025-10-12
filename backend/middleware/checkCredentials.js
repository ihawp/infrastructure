import authModels from '../models/authModels.js';
import bcrypt from 'bcrypt';

const checkCredentials = async (req, res, next) => {

    const { email, password } = req.body;

    const user = await authModels.login(email);

    if (!user) {
        return res
            .status(401)
            .json({
                success: false,
                message: 'Invalid credentials'
            });
    }

    if (!bcrypt.compare(password, user.password)) {
        return res
            .status(401)
            .json({
                success: false,
                message: 'Invalid credentials'
            });
    }

    next();

}

export default checkCredentials;