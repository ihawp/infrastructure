import authModels from '../models/authModels.js';
import bcrypt from 'bcrypt';

const checkCredentials = async (req, res, next) => {
    const { email, password } = req.body;

    try {
        const users = await authModels.login(email);

        // Check if user exists and get first result
        if (!users || users.length === 0) {
            return res.status(401).json({
                success: false,
                message: 'Invalid credentials'
            });
        }

        const user = users[0];

        // Check if password matches
        const isValidPassword = await bcrypt.compare(password, user.password);

        if (!isValidPassword) {
            return res.status(401).json({
                success: false,
                message: 'Invalid credentials'
            });
        }

        // Add user info to request for use in controller
        req.user = {
            id: user.id,
            username: user.username,
            email: email
        };

        next();
    } catch (error) {
        console.error('Error in checkCredentials:', error);
        return res.status(500).json({
            success: false,
            message: 'Internal server error'
        });
    }
};

export default checkCredentials;
