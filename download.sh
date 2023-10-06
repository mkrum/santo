
YEAR=$1

GL_ZIP=gl${YEAR}.zip
EV_ZIP=${YEAR}eve.zip

mkdir data/${YEAR}

pushd data/${YEAR}
wget https://www.retrosheet.org/gamelogs/$GL_ZIP
unzip $GL_ZIP

wget https://www.retrosheet.org/events/$EV_ZIP
unzip $EV_ZIP

echo rm $GL_ZIP
echo rm $EV_ZIP

popd
