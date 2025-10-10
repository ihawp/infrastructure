import db from '../utils/db.js';

const authModels = {
    registerModel: async (name, firstname, lastname) => {
        name = 'bananaphone';
        firstname = 'warren';
        lastname = 'chemerika';

        const [response] = await db.query(
            `
            INSERT INTO users 
            (username, firstname, lastname) 
            VALUES (?, ?, ?)`,
            [name, firstname, lastname]
        );
        return response;
    },
    loginModel: (req, res) => {
        res.send('hello world');
    },
    magicLinkModel: (req, res) => {
        res.send('hello world');
    }
};

export default authModels;
