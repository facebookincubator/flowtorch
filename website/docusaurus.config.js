const math = require('remark-math');
const katex = require('rehype-katex');

module.exports = {
  title: 'FlowTorch',
  tagline: 'Easily learn and sample complex probability distributions with PyTorch',
  url: 'https://flowtorch.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'facebookincubator',
  projectName: 'flowtorch',
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X',
      crossorigin: 'anonymous',
    },
  ],
  baseUrlIssueBanner: true,
  themeConfig: {
    gtag: {
      trackingID: 'G-X921WXL42L',
      anonymizeIP: true,
    },
    announcementBar: {
      id: 'supportus',
      content:
        '⭐️ If you like FlowTorch, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/facebookincubator/flowtorch">GitHub</a>! ⭐️',
    },
    prism: {
      theme: require("prism-react-renderer/themes/github"),
      darkTheme: require("prism-react-renderer/themes/dracula"),
    },
    navbar: {
      title: 'FlowTorch',
      logo: {
        alt: 'FlowTorch Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: 'users',
          activeBasePath: 'users',
          label: 'Users',
          position: 'left',
        },
        {
          to: 'dev',
          activeBasePath: 'dev',
          label: 'Developers',
          position: 'left',
        },
        {
          to: 'api',
          activeBasePath: 'api',
          label: 'Reference',
          position: 'left',
        },
        {
          href: 'https://github.com/facebookincubator/flowtorch/discussions',
          label: 'Discussions',
          position: 'right',
        },
        {
          href: 'https://github.com/facebookincubator/flowtorch/releases',
          label: 'Releases',
          position: 'right',
        },
        {
          href: 'https://github.com/facebookincubator/flowtorch',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Users Guide',
              to: 'users',
            },
            {
              label: 'Developers Guide',
              to: 'dev',
            },
            {
              label: 'API Reference',
              to: 'api',
            },
            {
              label: 'Roadmap',
              href: 'https://github.com/facebookincubator/flowtorch/projects',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Raise an issue',
              href: 'https://github.com/facebookincubator/flowtorch/issues/new/choose',
            },
            {
              label: 'Ask for help',
              href: 'https://github.com/facebookincubator/flowtorch/discussions/new',
            },
            {
              label: 'Give us feedback',
              href: 'https://github.com/facebookincubator/flowtorch/discussions/categories/feedback',
            },
            {
              label: 'Fork the repo',
              href: 'https://github.com/facebookincubator/flowtorch/fork',
            },
          ],
        },
        {
          title: 'Legal',
          items: [
            {
              label: 'MIT Open Source License',
              href: 'https://github.com/facebookincubator/flowtorch/blob/main/LICENSE.txt',
            },
            {
              label: 'Code of Conduct',
              href: 'https://www.contributor-covenant.org/version/1/4/code-of-conduct/',
            },
	    // Please do not remove the privacy and terms, it's a legal requirement.
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc. and its affiliates. All Rights Reserved.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/facebookincubator/flowtorch/edit/main/website/',
          routeBasePath: '/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
