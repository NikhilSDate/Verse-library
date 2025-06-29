#!/bin/bash
sudo docker build ../ -t insulinpump
sudo enroot import dockerd://insulinpump 