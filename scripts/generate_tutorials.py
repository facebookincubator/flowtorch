# Copyright (c) Meta Platforms, Inc

"""
Generates MDX (Markdown + JSX, see https://mdxjs.com/) files and sidebar
information for the Docusaurus v2 website from the Jupyter notebooks
in `/tutorials`.

"""

import errno
import os
import toml

from flowtorch.tutorials.tutorial import Tutorial


if __name__ == "__main__":
    # Load and validate configuration file
    import flowtorch

    main_path = flowtorch.__path__[0]
    config_path = os.path.join(main_path, "../website/tutorials.toml")
    config = toml.load(config_path)

    # Create directories if they don't exist
    sidebar_path = os.path.join(main_path, config["paths"]["sidebar"])
    markdown_path = os.path.join(main_path, config["paths"]["markdown"])

    # TODO: Factor out this function to flowtorch.utils
    def create_paths(path: str) -> None:
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    create_paths(sidebar_path)
    create_paths(markdown_path)

    # Extract list of tutorials and check whether exist
    excepts_paths = []
    #try:
    tutorials = [Tutorial(**kwargs) for kwargs in config['tutorial']]
    #except:
    #    pass

    # Check that all id's in sidebar exist and are unique
    excepts_ids = []
    # TODO

    # Display errors thus far
    # TODO
    
    # Extract menu hierarchy and build tutorial sidebar
    # TODO
    print(config['section'])

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
