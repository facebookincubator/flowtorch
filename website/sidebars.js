/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

 var apiSideBar = require('./api.sidebar.js');
 var tutorialSideBar = require('./tutorial.sidebar.js');

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  usersSidebar: {
    "Getting Started": ['users/introduction', 'users/installation', 'users/start'],
    "Tutorials": ['users/univariate', 'users/multivariate'],
    "Basic Concepts": ['users/shapes'],
  },
  devsSidebar: {
    "General": ['dev/contributing', 'dev/releases', 'dev/about'],
    "Extending the Library": ['dev/overview', 'dev/ops', 'dev/docs', 'dev/tests'],
    "Resources": ['dev/bibliography'],
  },
  apiSidebar: apiSideBar,
  tutorialSidebar: tutorialSideBar,
};

module.exports = sidebars;
