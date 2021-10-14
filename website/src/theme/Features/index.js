import React from "react";
import clsx from "clsx";
import { FaMeteor, FaDumbbell, FaHandsHelping, FaCubes, FaIndustry } from "react-icons/fa";

import styles from "./styles.module.css";

const size = 24;
const data = [
  {
    icon: <FaMeteor size={size} />,
    title: <>Simple but powerful</>,
    description: (
      <>
        Design, train, and sample from complex probability distributions using only a few lines of code. Advanced features such as conditionality, caching, and structured representations are planned for future released.
      </>
    ),
  },
  {
    icon: <FaHandsHelping size={size} />,
    title: <>Community focused</>,
    description: (
      <>
        We help you be a successful user or contributor through detailed user, developer, and API guides. Educational tutorials and research benchmarks are planned for the future. We welcome your feedback!
      </>
    ),
  },
  {
    icon: <FaCubes size={size} />,
    title: <>Modular and extendable</>,
    description: (
      <>
        Combine multiple bijections to form complex normalizing flows, and mix-and-match conditioning networks with bijections.
        FlowTorch has a well-defined interface so you easily create your own components!
      </>
    ),
  },
  {
    icon: <FaIndustry size={size} />,
    title: <>Production ready</>,
    description: (
      <>
        Proven code migrated from Pyro, with improved unit testing and continuous integration.
        And it is easy to add standard unit tests to components you write yourself!
      </>
    ),
  },
];

function Feature({ icon, title, description }) {
  return (
    <div className={clsx("col col--6", styles.feature)}>
      <div className="item">
        <div className={styles.header}>
          {icon && <div className={styles.icon}>{icon}</div>}
          <h2 className={styles.title}>{title}</h2>
        </div>
        <p>{description}</p>
      </div>
    </div>
  );
}

function Features() {
  return (
    <>
      {data && data.length && (
        <section id="features" className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <div className="row">
                  {data.map((props, idx) => (
                    <Feature key={idx} {...props} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </>
  );
}

export default Features;
