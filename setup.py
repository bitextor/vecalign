#!/usr/bin/env python

import setuptools

def reqs_from_file(src):
    requirements = []

    with open(src) as f:
        for line in f:
            line = line.strip()

            if not line.startswith("-r"):
                requirements.append(line)
            else:
                add_src = line.split(' ')[1]
                add_req = reqs_from_file(add_src)
                requirements.extend(add_req)

    return requirements

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    requirements = reqs_from_file("requirements.txt")

    setuptools.setup(
        name="vecalign",
        version="1.2",
        install_requires=requirements,
        license="Apache License 2.0",
        description="Improved Sentence Alignment in Linear Time and Space",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/bitextor/vecalign",
        packages=["vecalign"],
        package_data={"vecalign": ["dp_core.pyx"]},
        entry_points={
            "console_scripts": [
                "vecalign = vecalign.vecalign:_main",
                "vecalign-overlap = vecalign.overlap:_main"
            ]
        }
        )
