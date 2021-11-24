/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
