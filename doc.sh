# Build docs, copy to correct docs folder, delete build
cd docs/src
sphinx-apidoc -o ./doc ../../matid
make html
cp -a build/html/. ../
rm -r build
