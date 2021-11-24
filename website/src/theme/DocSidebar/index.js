import OriginalDocSidebar from '@theme-original/DocSidebar';
import React from 'react';


export default function DocSidebar(props) {
  return (
    <>
      <div id={props.path}>
      <OriginalDocSidebar {...props} />
      </div>
    </>
  );
}