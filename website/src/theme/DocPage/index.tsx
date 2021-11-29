/**
 * Copyright (c) Meta Platforms, Inc
 */

import React from 'react';

import OriginalDocPage from '@theme-original/DocPage';
import NotFound from '@theme/NotFound';
import type {Props} from '@theme-original/DocPage';
import {matchPath} from '@docusaurus/router';


function DocPage(props: Props): JSX.Element {
  const {
    route: {routes: docRoutes},
    versionMetadata,
    location,
  } = props;
  const currentDocRoute = docRoutes.find((docRoute) =>
    matchPath(location.pathname, docRoute),
  );
  if (!currentDocRoute) {
    return <NotFound />;
  }
  return (
    <>
      <div id={'docpage'+currentDocRoute.path}>
      <OriginalDocPage {...props} />
      </div>
    </>
  );
}

export default DocPage;
