# Build docs, copy to correct docs folder, delete build
git checkout master
cd src
sphinx-apidoc -o ./doc ../../matid
make html
cp -a build/html/. ../
rm -r build
