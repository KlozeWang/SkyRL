docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  --privileged \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --cgroupns=host \
  --user root \
  -v ~/xujiawang/SkyRL:/workspace \
  --name skyrl0410 \
  skyrl:0409 \
  /bin/bash
