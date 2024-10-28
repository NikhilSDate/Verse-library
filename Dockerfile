FROM python:3.11-bookworm

# do everything in a virtual env
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies
COPY requirements.txt /tmp/verse_requirements.txt
RUN pip install -r /tmp/verse_requirements.txt

# so that we can export images
RUN pip install kaleido

# libgl1 needed for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-dev

# add the verse python library
RUN mkdir /opt/verse
WORKDIR /opt/verse
COPY requirements.txt /opt/verse/requirements.txt
COPY setup.py /opt/verse/setup.py
COPY verse /opt/verse/verse
RUN pip install /opt/verse
