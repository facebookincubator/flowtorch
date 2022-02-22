# Copyright (c) Meta Platforms, Inc

"""
Generates MDX (Markdown + JSX, see https://mdxjs.com/) files and sidebar
information for the Docusaurus v2 website from the Jupyter notebooks
in `/tutorials`.

"""

import toml
from flowtorch.tutorials.tutorial import Tutorial


if __name__ == "__main__":
    # Extract list of tutorials and check whether exist
    with open('tutorials.toml', 'r') as f:
        tutorial_settings = toml.load(f)
    excepts_paths = []
    #try:
    tutorials = [Tutorial(**kwargs) for kwargs in tutorial_settings['tutorial']]
    #except:
    #    pass

    # Check that all id's in sidebar exist and are unique
    excepts_ids = []
    # TODO

    # Display errors thus far
    # TODO
    
    # Extract menu hierarchy and build tutorial sidebar
    # TODO
    print(tutorial_settings['section'])

    # Build index page
    # TODO

    # Temporary save empty MDX files for tutorials

    # **********************************
    # Tuesday
    # **********************************
    # Convert tutorials to MDX
    # TODO

    # Afterwards
    # * Integrate existing three tutorials
    # * Create a PR for this
    # * TSC tutorial
    # * API generation system
    # * Roadmap for milestones, and send to JP, Vincent
    # * Miles in tasks

    # tutorials_metadata = load_nbs_to_convert()
    # print("Converting tutorial notebooks into mdx files")
    
    """
    for _, value in tutorials_metadata.items():
        path = Path(value["nb_path"]).resolve()
        print(f"{path.stem}")
        mdx, jsx = transform_notebook(path)
    print("")
    """
