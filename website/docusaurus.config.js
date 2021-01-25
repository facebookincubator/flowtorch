module.exports = {
  title: 'FlowTorch',
  tagline: 'A PyTorch library for flexible high-dimensional probability distributions.',
  url: 'https://flowtorch.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'stefanwebb', // Usually your GitHub org/user name.
  projectName: 'flowtorch', // Usually your repo name.
  themeConfig: {
    announcementBar: {
      id: 'supportus',
      content:
        '⭐️ If you like FlowTorch, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/stefanwebb/flowtorch">GitHub</a>! ⭐️',
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
          activeBasePath: 'docs',
          label: 'Users',
          position: 'left',
        },
        {
          to: 'dev',
          activeBasePath: 'docs',
          label: 'Developers',
          position: 'left',
        },
        {
          to: 'api',
          activeBasePath: 'docs',
          label: 'Reference',
          position: 'left',
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
              label: 'API Reference',
              to: 'api',
            },
            {
              label: 'Users Guide',
              to: 'users',
            },
            {
              label: 'Developers Guide',
              to: 'dev',
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
              label: 'GitHub',
              href: 'https://github.com/stefanwebb/flowtorch',
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
          // Please change this to your repo.
          editUrl:
            'https://github.com/stefanwebb/flowtorch/edit/master/website/',
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
