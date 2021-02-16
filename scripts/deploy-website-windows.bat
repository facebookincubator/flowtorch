cd ..
git checkout master
rmdir /q /s .website
cd website
yarn install
yarn build
move ./build ../.website
cd ..
git checkout website
git clean -f -d
git rm -r .
xcopy /E /I .website ./
git add .
git commit -a -m "Updating website"
git push
git checkout master
rmdir /q /s .website