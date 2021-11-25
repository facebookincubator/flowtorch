import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {MDXProvider} from '@mdx-js/react';
import MDXComponents from '@theme/MDXComponents';


import styles from "./styles.module.css";

function PythonMethod({children}) {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
  <div
    className='doc-method'
    style={{
      backgroundColor: 'DarkSalmon',
      borderRadius: '2px',
      color: '#fff',
      padding: '16px',
      borderRadius: '0.5em',
    }}>

    <div style={{fontSize: '125%'}}>
    
    {children}

    </div>
  </div>
  );
}


/*

<div class="doc-symbol doc-class">
  <div class="">
    <h5> </h5>

</div>

*/

export default PythonMethod;
