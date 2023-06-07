install-test:
	docker build -t miplearnjl-test -f test/docker/Dockerfile .
	docker run --rm -it miplearnjl-test