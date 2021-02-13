import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import styles from "./styles.module.css";

function Hero() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
    <header id="hero" className={clsx("hero", styles.banner)}>
      <div className="container">
          <div className="row">
            
          <h1 className="hero__subtitle">
            <img className="hero__img" src="img/logo.svg" />
            Easily <span className="hero__primary">learn</span> and <span className="hero__primary">sample</span> complex <span className="hero__secondary">probability distributions</span> with PyTorch
          </h1>

          </div>
          <div className="hero__buttons row">
            <div className={styles.buttons}>
              <Link
                className={clsx(
                  "button button--primary button--lg",
                  styles.getStarted
                )}
                to={useBaseUrl("users")}
              >
                Get Started
              </Link>
            </div>
            <div className={styles.buttons}>
              <Link
                className={clsx(
                  "button button--warning button--lg",
                  styles.getStarted
                )}
                to={useBaseUrl("dev")}
              >
                Contribute
              </Link>
              
                <iframe
                  className="hero__github_button"
                  src="https://ghbtns.com/github-btn.html?user=stefanwebb&amp;repo=flowtorch&amp;type=star&amp;count=true&amp;size=large"
                  width={160}
                  height={30}
                  title="GitHub Stars"
                />
              
            </div>
          </div>
        </div>
    </header>
  );
}

export default Hero;
