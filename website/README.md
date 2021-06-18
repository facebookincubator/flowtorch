<a href="https://flowtorch.ai"><img src="https://github.com/facebookincubator/flowtorch/raw/master/gh-pages/flowtorch-ai.png" width="100%" /></a>

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator, and hosted using GitHub pages at [https://flowtorch.ai](https://flowtorch.ai). The source for the website is located in [master/website](https://github.com/facebookincubator/flowtorch/tree/master/website) and it is hosted from the root directory of the [website](https://github.com/facebookincubator/flowtorch/tree/website) branch.

## Preparation
1. Install [Node.js](https://nodejs.org/).
2. Install [Yarn](https://yarnpkg.com/):
```console
npm install --global yarn
```
4. Navigate to [master/website](https://github.com/facebookincubator/flowtorch/tree/master/website) and install the dependencies:
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

This command generates static content into the `website/build` directory, which is deployed by copying into the [gh-pages](https://github.com/facebookincubator/flowtorch/tree/gh-pages) branch.

## Deployment

Core developers can deploy the website as follows:

```console
GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

Activity logs for all past deployments to GitHub pages can be viewed [here](https://github.com/facebookincubator/flowtorch/deployments/activity_log?environment=github-pages). For your convenience, there is [a script to deploy the website on Windows](https://github.com/facebookincubator/flowtorch/tree/master/deploy-website-windows.bat).
