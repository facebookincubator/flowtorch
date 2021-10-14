var apiSideBar = require('./api.sidebar.js');

module.exports = {
  usersSidebar: {
    "Getting Started": ['users/introduction', 'users/installation', 'users/start'],
    "Normalizing Flows": ['users/univariate', 'users/multivariate', 'users/conditional'],
    "Basic Concepts": ['users/shapes'],
  },
  devsSidebar: {
    "General": ['dev/contributing', 'dev/releases', 'dev/about'],
    "Extending the Library": ['dev/overview', 'dev/ops', 'dev/bijector', 'dev/params', 'dev/docs', 'dev/tests'],
    "Resources": ['dev/bibliography'],
  },
  apiSidebar: apiSideBar,
};
