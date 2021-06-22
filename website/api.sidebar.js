module.exports = [
    'api/overview',
    {
      type: 'category',
      label: 'flowtorch',
      collapsed: true,
      items: ['api/flowtorch', 'api/flowtorch.Bijector', 'api/flowtorch.Params', 'api/flowtorch.ParamsModule'],
    },
    {
      type: 'category',
      label: 'flowtorch.bijectors',
      collapsed: true,
      items: ['api/flowtorch.bijectors', 'api/flowtorch.bijectors.Affine'],
    },
];
