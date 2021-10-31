import React from "react";
import CodeSnippet from "@theme/CodeSnippet";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

import Headline from "@theme/Headline";
import snippets from "./snippets";
import styles from "./styles.module.css";

function renderTabs() {
  return (
    <Tabs
      defaultValue={snippets[0].label}
      values={snippets.map((props, idx) => {
        return { label: props.label, value: props.label };
      })}
      className={styles.tabs}
    >
      {snippets.map((props, idx) => (
        <TabItem key={idx} value={props.label}>
          <CodeSnippet key={idx} {...props} />
        </TabItem>
      ))}
    </Tabs>
  );
}

function Examples() {
  return (
    <>
      {snippets && snippets.length && (
        <section id="examples" className={styles.examples}>
          <div className="container">
            <div className="row">
              <div className="col col--6">
                <Headline
                  category="Examples"
                />
                {renderTabs()}
              </div>
              <div className="col col--6">
              <div className={styles.example_container}>
              <svg className={styles.animation_svg} xmlns="http://www.w3.org/2000/svg">
              <g className={styles.animation}>
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-0.svg" />
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-1.svg" />
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-2.svg" />
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-3.svg" />
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-4.svg" />
              <image height="30rem" height="30rem" href="img/bivariate-normal-frame-5.svg" />
              </g>
              </svg>
              </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </>
  );
}

export default Examples;
