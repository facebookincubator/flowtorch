import OriginalDocPage from '@theme-original/DocPage';
import NotFound from '@theme/NotFound';
import React from 'react';
import {matchPath} from '@docusaurus/router';

function DocPage(props) {
  const {
    route: {routes: docRoutes},
    versionMetadata,
    location,
  } = props;
  const currentDocRoute = docRoutes.find((docRoute) =>
    matchPath(location.pathname, docRoute),
  );
  if (!currentDocRoute) {
    return (
      <>
        <div id={'notfound'+currentDocRoute.path}>{props.path}
        <NotFound {...props} />;
        </div>
      </>
    );
  }
  return (
    <>
      <div id={'docpage'+currentDocRoute.path}>{props.path}
      <OriginalDocPage {...props} />
      </div>
    </>
  );
}

export default DocPage;