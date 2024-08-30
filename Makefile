local_build:
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt

docker_build:
	docker build --tag f_seg_comp .