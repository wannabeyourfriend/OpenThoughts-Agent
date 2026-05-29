
If I change path in file sharing in Docker and then run in Terminal: DOCKER_CONTENT_TRUST=1 docker run -v /volumes/Drive -p 8000:8000 obsidiansystems/tezos-bake-monitor:0.4.0 --pg-connection="host=host.docker.internal port=5432 dbname=postgres user=postgres password=my password ” Will Kiln be installed on /volumes/Drive?
