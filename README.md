I am learning about configuring MySQL and Redis clusters.

Locally, I have a Redis cluster with 6 nodes (3 masters, 3 replicas) and an InnoDB cluster with 2 nodes (1 primary, 1 replica).

The infrastructure consists of a server for reading/writing to the database and Redis, and a set of workers that process jobs from Redis. I use the BullMQ library to implement a Job Queue in Node.js, where each instance manages its own jobs and correctly adds or removes data in Redis.

The workers are separate instances hosted on another machine. They read jobs on their own schedule and complete them, returning a response such as 'completed' or 'failed'. These responses determine whether a job needs to be retried or reassigned.

Most of my previous personal projects run on shared hosting with minimal infrastructure concerns, just a basic Apache server capable of running Python and Node.js. This is my first time intentionally designing and managing robust infrastructure, including Redis and MySQL clusters, job queues, and multi-node worker orchestration.

I had purchased a VPS in the summer, and did some experimentation with setting up firewalls and databases, but then I had other things to do :).

[]()