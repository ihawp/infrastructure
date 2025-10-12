const magicKey = {
    send: async function(job) {
        console.log('Sending magic key for job:', job.data);
        return 'Magic key sent';
    }
}

export default magicKey;