var apiSideBar = require('./api.sidebar.js');

module.exports = {
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
};
