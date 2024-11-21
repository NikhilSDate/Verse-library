# Thermometer modeling

Started October 2024.

## Development environment

A dockerfile is provided in the root of the repository. Mount the verse repo to `/opt/verse` instead of copying it in the dockerfile for faster refreshes.

```sh
docker build -t verse/test .
docker run -it -v $PWD:/opt/verse --rm verse/test /bin/bash
```

To view images externally, spin up a local `python3 -m http.server 8000` and forward over ssh with the following command (run on the client you are ssh-ing into this server from):
```sh
ssh -N -L 8000:localhost:8000 vector0.sprai.org
```

## Images in headless environments

Instead of `fig.show()`, use `fig.write_image('out.png')` in headless environments.
