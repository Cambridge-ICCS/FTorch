# Codespace for FTorch

If you want to try out FTorch from your browser in a configured environment
you can use a [GitHub Codespace](https://github.com/features/codespaces).
This allows you to work in a VSCode Web session running code on from a container in
the cloud.


## Getting Started

All GitHub users have a certain number of hours of credit for using codespaces.
This can be extended if you have GitHub Education.

To launch a codespace for FTorch navigate to the
[repository on Github](https://github.com/Cambridge-ICCS/FTorch) and click the down
arrow on the 'code button'.
Select 'Codespaces' and then 'Create codespace on main'.
This will open a new window and start up the interactive VSCode session.

Once the container is set up you will be dropped into a VSCode session with
dependencies installed and a copy of the FTorch repository cloned.

> [!NOTE]  
> Firefox users with enhanced tracking protection will need to disable this for
> the codespace.


## Installing FTorch

To install FTorch you can execute the script in the `codespace/` directory from the
terminal:

```sh
source codespace/build_FTorch.sh
```

> [!NOTE]  
> By using `source` this will run in your current terminal, allowing the modified
> `LD_LIBRARY_PATH` to persist and leaving in the virtual environment with `torch`
> installed.


## Examples

You can now navigate to the examples and build them following the instructions in their
respective `README`s.

Note:

- You do not need to create a new virtual environment for each exercise, instead use
  the one that is active following installation above.
- You should link to the FTorch installation at `/workspaces/FTorch/bin/` when
  specifying a `CMAKE_PREFIX_PATH`.


## Setup

The codespace configuration is defined in the `.devcontainer/` directory under the root
of FTorch.
This contains a `Dockerfile` defining the Docker container we will work in - a base
container with CMake, gfortran, and python installed - and `devcontainer.json` defining
the codespace.
