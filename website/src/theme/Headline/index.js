import React from "react";
import { PropTypes } from "prop-types";

import styles from "./styles.module.css";

function Headline(props) {
  const { category, title, subtitle, offset } = props;

  return (
    <div className="row">
      <div className={`col col--${12 - offset} col--offset-${offset}`}>
        <div className={styles.headline}>
          {category && <span className={styles.category}>{category}</span>}
          {title && <h2 className={styles.title}>{title}</h2>}
          {subtitle && <h3 className={styles.subtitle}>{subtitle}</h3>}
        </div>
      </div>
    </div>
  );
}

Headline.propTypes = {
  category: PropTypes.string,
  title: PropTypes.string,
  subtitle: PropTypes.string,
  offset: PropTypes.number,
};

Headline.defaultProps = {
  offset: 0,
};

export default Headline;
