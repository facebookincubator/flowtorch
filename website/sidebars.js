var apiSideBar = require('./api.sidebar.js');

module.exports = {
  usersSidebar: {
    "Getting Started": ['users/introduction', 'users/installation', 'users/start'],
    "Normalizing Flows": ['users/univariate', 'users/multivariate', 'users/conditional', 'users/methods'],
    "Basic Concepts": ['users/shapes', 'users/constraints', 'users/bijectors', 'users/parameters', 'users/transformed_distributions', 'users/conditioning', 'users/composing', 'users/gpu_support', 'users/serialization'],
    "Advanced Topics": ['users/caching', 'users/initialization', 'users/structure', 'users/torchscript'],
  },
  devsSidebar: {
    "General": ['dev/contributing', 'dev/releases', 'dev/about'],
    "Extending the Library": ['dev/overview', 'dev/ops', 'dev/bijector', 'dev/params', 'dev/docs', 'dev/tests'],
    "Resources": ['dev/bibliography'],
  },
  apiSidebar: apiSideBar,
};
