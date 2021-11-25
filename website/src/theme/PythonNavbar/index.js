import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {MDXProvider} from '@mdx-js/react';
import MDXComponents from '@theme/MDXComponents';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCode, faChevronRight, faTerminal, faAngleDoubleRight } from '@fortawesome/free-solid-svg-icons'

import styles from "./styles.module.css";

function PythonNavbar({children, url}) {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
  <div className='doc-navbar'
      style={{
    //backgroundColor: "#eeeeee",
    //backgroundColor: unset,
    borderRadius: '2px',
    color: '#fff',
    padding: '16px',
    borderRadius: '0.5em',
    marginBottom: '1rem'
  }}>
    <div className='doc-navbar-links'>
      <span style={{fontFamily: "Consolas, monospace"}}>
      {children}
      </span>
    </div>    
    <div className='doc-navbar-url'><a href={url} target="_blank"><FontAwesomeIcon icon={faTerminal} /></a></div>
  </div>
  );
}


/*

<div class="doc-symbol doc-class">
  <div class="">
    <h5> </h5>

</div>

*/

export default PythonNavbar;
