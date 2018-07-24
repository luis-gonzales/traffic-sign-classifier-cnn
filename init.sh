cd data
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
unzip traffic-signs-data.zip
rm traffic-signs-data.zip
cd ..
cd tf_model
gdrive download 1E2Z1zr6lTUK5wcuc4BurCQGA0Ea9cP9g
unzip tf_saved_model.zip
rm tf_saved_model.zip
cd ..
