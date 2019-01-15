# Build docs, copy to correct docs folder, delete build
git checkout master
cd docs/src
sphinx-apidoc -o ./doc ../../matid
make html
cp -a build/html/. ../
rm -r build

# Push changes to docs
git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"
cd ../..
git add ./docs
git commit -m "Travis documentation build: $TRAVIS_BUILD_NUMBER"
git push --quiet https://SINGROUP:$GH_TOKEN@github.com/SINGROUP/matid master &>/dev/null
