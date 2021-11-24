import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {MDXProvider} from '@mdx-js/react';
import MDXComponents from '@theme/MDXComponents';


import styles from "./styles.module.css";

function PythonClass({children}) {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
  <div
    className='doc-class'
    style={{
      backgroundColor: "#25c2a0",
      borderRadius: '2px',
      color: '#fff',
      padding: '16px',
      borderRadius: '0.5em',
      marginBottom: '1.0rem',
    }}>

    <div style={{fontSize: '150%'}}>
    
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

export default PythonClass;
