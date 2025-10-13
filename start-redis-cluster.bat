@echo off

redis-cli FLUSHALL -a w

start "Redis-7000" redis-server redis-cluster/7000/7000.conf
start "Redis-7001" redis-server redis-cluster/7001/7001.conf
start "Redis-7002" redis-server redis-cluster/7002/7002.conf
start "Redis-7003" redis-server redis-cluster/7003/7003.conf
start "Redis-7004" redis-server redis-cluster/7004/7004.conf
start "Redis-7005" redis-server redis-cluster/7005/7005.conf

timeout /t 5 /nobreak >nul

echo yes | redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 --cluster-replicas 1 -a w

pause