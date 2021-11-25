import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {MDXProvider} from '@mdx-js/react';
import MDXComponents from '@theme/MDXComponents';


import styles from "./styles.module.css";

function PythonModule({children}) {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
  <div
    className='doc-module'
    style={{
      backgroundColor: "CornflowerBlue",
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

export default PythonModule;
