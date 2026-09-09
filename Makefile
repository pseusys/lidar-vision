.ONESHELL:
.EXPORT_ALL_VARIABLES:
.DEFAULT_GOAL := help

SHELL = /bin/bash

# Ubuntu (this project's CI runner, and most Linux) only ships `python3`, so
# prefer it there. On Windows, `python3` is a trap: it resolves to the
# Microsoft Store's install stub, which `command -v` finds even though
# running it does nothing useful — see memory/gotchas.md — so prefer bare
# `python` there instead, and only fall back to `python3` if that's missing.
ifeq ($(OS),Windows_NT)
PYTHON := $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
VENV_BIN := venv/Scripts
else
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
VENV_BIN := venv/bin
endif
VENV_PYTHON := $(VENV_BIN)/python
PATH := $(VENV_BIN):$(PATH)

ROBAIR_IP=192.168.1.201
CONTAINER_IP := $(shell hostname -I | awk '{print $$1}')


help:
	@ # Display help information
	echo "Welcome to the 'lidar-vision' project!"
	echo "This is the available make command list:"
	echo "    'make venv': create python virtual environment and install latest jupyter server libraries as well as follow_the_drow library, enable required params."
	echo "    'make build-lib': build follow_the_drow C++ lib, install the lib if run as superuser."
	echo "    'make test': run project-internals unit tests (tests/, pytest, CPU-only)."
	echo "    'make build-image': build required Docker image locally and test successful creation."
	echo "    'make launch-docker-local': launch ROS pipeline in docker on local device, don't even ty to connect to any robot."
	echo "    'make launch-docker-robot': test RobAIR connection and run ROS pipeline on the RobAIR."
	echo "        WARNING: the default RobAIR IP address is 192.168.1.201; to use another IP use the following syntax: 'make launch-docker-robot ROBAIR_IP=X.X.X.X'."
	echo "    'make clean-docker': clean all Docker related artifacts, stop container and remove image, that is useful because Docker image caches are HUGE."
	echo "    'make clean-local': remove all file artifacts in this repo (including installed follow_the_drow C++ library if any)."
	echo "    'make clean': remove all project traces from the computer."
.PHONY: help



venv:
	@ # Create python virtual environment
	$(PYTHON) -m venv venv
	$(VENV_PYTHON) -m pip install --upgrade pip jupyter~=1.0
	$(VENV_PYTHON) -m pip install -e library
	$(VENV_PYTHON) -m pip install -r utils/requirements.txt

build-lib:
	@ # Build and install `follow_the_drow` library locally
	mkdir -p library/cpp_core/build
	cd library/cpp_core/build
	cmake ..
	make
	test $$(id -u) -eq 0 && { make install; } || { echo "Installation can be performed in superuser mode only!"; }
.PHONY: build-lib



test: venv
	@ # Run project-internals unit tests (algorithms/geometry, no GPU or dataset needed)
	$(VENV_PYTHON) -m pytest tests -v
.PHONY: test



build-image:
	@ # Build docker image - internet connection required!
	docker compose -f deploy/docker/docker-compose.yml build
	test -n "$$(docker images -q follow_the_drow 2> /dev/null)" || { echo "Image building error!"; exit 1; }
.PHONY: build-image

launch-docker-local:
	export ROBAIR_IP=127.0.0.1
	docker compose -f deploy/docker/docker-compose.yml up --force-recreate roslaunch
.PHONY: launch-docker-local

launch-docker-robot:
	ping -q -W 1 -c 1 $(ROBAIR_IP) 2>&1 > /dev/null || { echo "Not connected to RobAIR network!"; exit 1; }
	docker compose -f deploy/docker/docker-compose.yml up --force-recreate roslaunch
.PHONY: launch-docker-robot



report:
	@ # Compile report LaTeX
	docker run --rm -i -v $(CURDIR)/report:/workdir arkark/latexmk /bin/bash -c "latexmk -pdf main.tex && chown -R '$(shell id -u):$(shell id -g)' /workdir"
.PHONY: report



clean-docker:
	docker rmi follow_the_drow
	docker rm -f ros-desktop-follow-the-drow
.PHONY: clean-docker

clean-local:
	if [ -f library/cpp_core/build/install_manifest.txt ]; then
		xargs rm < library/cpp_core/build/install_manifest.txt
	fi
	rm -rf venv
	rm -rf library/cpp_core/build
	rm -rf library/follow_the_drow.egg-info
	rm -rf library/follow_the_drow/*.so
	rm -rf library/follow_the_drow/include
	rm -rf deploy/out
	rm -rf **/__pycache__
	rm -rf report/out
.PHONY: clean-local

clean: clean-docker clean-local
.PHONY: clean
