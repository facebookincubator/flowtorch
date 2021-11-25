import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// Our theme
import Examples from "@theme/Examples";
import Features from "@theme/Features";
import Hero from "@theme/Hero";


export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="FlowTorch is a library for representing complex or high-dimensional probability distributions.">
      <Hero />
      <main>
          <Features />
          <Examples />
      </main>
    </Layout>
  );
}
