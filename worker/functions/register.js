const register = {
    registerUser: async function(user) {
        console.log('Registering user', user);
        logger.info({ verificationEmail: 'sent', user });
        console.log('User registered');
        return true;
    }
}

export default register;