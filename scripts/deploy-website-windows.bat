git checkout master
rmdir /q /s .website
cd website
yarn build
move ./build ../.website
git checkout website
git rm -r .
cp -r .website/* .
git add .
git commit -a -m "[Commit message]"
git push
git checkout master