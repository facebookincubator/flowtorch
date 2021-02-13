module.exports = {
  title: 'FlowTorch',
  tagline: 'Easily learn and sample complex probability distributions with PyTorch',
  url: 'https://flowtorch.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'stefanwebb',
  projectName: 'flowtorch',
  baseUrlIssueBanner: true,
  themeConfig: {
    announcementBar: {
      id: 'supportus',
      content:
        '⭐️ If you like FlowTorch, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/stefanwebb/flowtorch">GitHub</a>! ⭐️',
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
          href: 'https://github.com/stefanwebb/flowtorch/discussions',
          label: 'Discussions',
          position: 'right',
        },
        {
          href: 'https://github.com/stefanwebb/flowtorch/releases',
          label: 'Releases',
          position: 'right',
        },
        {
          href: 'https://github.com/stefanwebb/flowtorch',
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
              href: 'https://github.com/stefanwebb/flowtorch/projects',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Raise an issue',
              href: 'https://github.com/stefanwebb/flowtorch/issues/new/choose',
            },
            {
              label: 'Ask for help',
              href: 'https://github.com/stefanwebb/flowtorch/discussions/new',
            },
            {
              label: 'Give us feedback',
              href: 'https://github.com/stefanwebb/flowtorch/discussions/categories/feedback',
            },
            {
              label: 'Fork the repo',
              href: 'https://github.com/stefanwebb/flowtorch/fork',
            },
          ],
        },
        {
          title: 'Legal',
          items: [
            {
              label: 'MIT Open Source License',
              href: 'https://github.com/stefanwebb/flowtorch/blob/master/LICENSE.txt',
            },
            {
              label: 'Code of Conduct',
              href: 'https://www.contributor-covenant.org/version/1/4/code-of-conduct/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} FlowTorch Development Team.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/stefanwebb/flowtorch/edit/master/website/',
          routeBasePath: '/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
