import bcrypt from 'bcrypt';
import queue from '../utils/bull';

const verifyAccess = async (req, res, next) => {

    const { accessToken } = req.signedCookies;

    console.log(accessToken);

    // is there an access token?
    if (!accessToken) {

        // try refresh token

        const { refreshToken } = req.signedCookies;

        if (!refreshToken) {
            return res.status(401).json({
                success: false,
                error: 'UNAUTHORIZED'
            });
        }

        // check database with refresh token.
        

    }

    // check redis for access token. Access tokens are stored for 15 minutes max.
    const redisToken = await queue.get(`accessToken:${req.user.id}`);

    if (!accessToken) {
        return res.status(401).json({
            success: false,
            error: 'UNAUTHORIZED'
        });
    }

    const accessTokenValid = await bcrypt.compare(accessToken, redisToken);

    if (!accessTokenValid) {
        return res
            .status(401)
            .json({
                success: false,
                error: 'UNAUTHORIZED',
                message: 'Invalid access token'
            });
    }

    next();

};

export default verifyAccess;
