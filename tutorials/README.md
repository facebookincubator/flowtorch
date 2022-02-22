This folder contains [Jupyter notebook](https://jupyter.org/) tutorials for [FlowTorch](https://flowtorch.ai). The `.ipynb` files are converted by [a script](https://github.com/facebookincubator/flowtorch/blob/main/scripts/generate_tutorials.py) to [MDX format](https://mdxjs.com/) for display [on the website](https://flowtorch.ai/tutorials).

## Creating a New Tutorial
To add a new tutorial to the library, follow these steps:
1. Discuss your proposal for a new tutorial on the [FlowTorch forum](https://github.com/facebookincubator/flowtorch/discussions).
2. Create a new branch on your local copy of your FlowTorch fork.
3. Add the new tutorial notebook to `/tutorials`.
4. Create an entry for the tutorial and add to the table of contents in [`/website/tutorials.toml`](https://github.com/facebookincubator/flowtorch/blob/main/website/tutorials.toml).
5. Test the addition by running `python scripts/generate_tutorials.py` and launching the website locally.
6. Commit your changes and create a new pull request based upon them.

## Generating and Viewing Markdown
The script [`scripts/generate_tutorials.py`](https://github.com/facebookincubator/flowtorch/blob/main/scripts/generate_tutorials.py) reads the configuration file [`website/tutorials.toml`](https://github.com/facebookincubator/flowtorch/blob/main/website/tutorials.toml) to generate markdown `website/docs/tutorials/*.mdx` and sidebar `website/tutorials.sidebar.js` for display on the website. This output is ephemeral, and, similar to the API docs, is git-ignored.

To test the display of tutorials, run the [above script](https://github.com/facebookincubator/flowtorch/blob/main/scripts/generate_tutorials.py) and launch the website locally using the [instructions here](https://github.com/facebookincubator/flowtorch/tree/main/website).