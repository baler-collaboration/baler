curl https://cernbox.cern.ch/remote.php/dav/public-files/21uZJO4hkqsQW6Z/baler.zip --output baler.zip
tar -xf baler.zip
cd baler
docker run --mount type=bind,source=%cd%/projects/,target=/baler-root/projects --mount type=bind,source=%cd%/data/,target=/baler-root/data pekman/baler:latest --project=example_CFD --mode=train
