<img src="https://github.com/stefanwebb/flowtorch/raw/master/website/flowtorch-ai.png" width="100%" />

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator, and hosted using GitHub pages at [https://flowtorch.ai](https://flowtorch.ai). The source for the website is located in [master/website](https://github.com/stefanwebb/flowtorch/tree/master/website) and it is hosted from the root directory of the [website](https://github.com/stefanwebb/flowtorch/tree/website) branch. 

## Preparation
1. Install [Node.js](https://nodejs.org/).
2. Install [Yarn](https://yarnpkg.com/):
```console
npm install --global yarn
```
3. Install [Docusaurus 2](https://v2.docusaurus.io/docs/installation).
4. Navigate to [master/website](https://github.com/stefanwebb/flowtorch/tree/master/website) and install the dependencies:
```console
cd website
yarn install
```

## Local Development

```console
yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

## Build

```console
yarn build
```

This command generates static content into the `website/build` directory, which is deployed by copying into the [website](https://github.com/stefanwebb/flowtorch/tree/website) branch.

## Deployment

Core developers can deploy the website as follows. On Linux/Mac, in the base directory:

> :exclamation: **The following commands for Linux/Mac have not yet been tested and debugged.**

```console
git checkout master
rm -r .website
cd website
yarn build
cp -r build/* ../.website
git checkout website
git rm -r .
cp -r .website/* .
git add .
git commit -a -m "Updating website"
git push
git checkout master
```

Or on Windows:
```console
git checkout master
rmdir /q /s website\build
cd website
yarn install
yarn build
cd ..
git checkout website
echo website/ > .gitignore
git clean -f -d
del .gitignore
git rm -r .
xcopy /E /I website\build .
git add .
git commit -a -m "Updating website"
git push
git checkout master
rmdir /q /s website\build
```

Activity logs for all past deployments to GitHub pages can be viewed [here](https://github.com/stefanwebb/flowtorch/deployments/activity_log?environment=github-pages). For your convenience, there is [a script to deploy the website on Windows](https://github.com/stefanwebb/flowtorch/tree/master/deploy-website-windows.bat).