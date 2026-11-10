---
substitutions:
  pic1: |-
    ```{image} https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/hstcombined.png?raw=true
    ```
  pic2: |-
    ```{image} https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/almacombined.png?raw=true
    ```
---

# What is PyAutoGalaxy?

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

{{ pic1 }}

**PyAutoGalaxy** also fits interferometer data from observatories such as ALMA:

{{ pic2 }}

# Getting Started

## Explore It Yourself

- **[Start Here](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_1_start_here.html)** — discover PyAutoGalaxy and work through your first example.
- **[Try it in your browser](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.11.9.1/start_here.ipynb)** — run the introductory notebook without installing anything.
- **[Install PyAutoGalaxy](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html)** — get set up on your computer.

Explore more examples in the [workspace](https://github.com/PyAutoLabs/autogalaxy_workspace), or learn galaxy modeling from the ground up with [HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy).

## Explore with AI

The [**PyAutoGalaxy AI Assistant**](https://github.com/PyAutoLabs/autogalaxy_assistant) lets you explore
galaxy structure and morphology through conversation. Ask a question, explore an idea, or describe the analysis
you want to run. Both routes use the same software. To get started, paste this prompt into Claude Code, Codex
or another AI coding agent:

:::{container} nl-prompt
> I want to study galaxy structure and morphology with PyAutoGalaxy (https://github.com/PyAutoLabs/PyAutoGalaxy)
> and the autogalaxy_assistant (https://github.com/PyAutoLabs/autogalaxy_assistant). First clone that
> repository, cd into it and follow its AGENTS.md.
>
> Begin the "start here" guide for a new user.
:::

The assistant guides you through [Start Here](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_1_start_here.html)
as a conversation, then helps you analyse your own galaxies. See [AI Options](https://github.com/PyAutoLabs/PyAutoScientist/blob/main/AI_OPTIONS.md)
for recommended tools, costs and free alternatives.

# Core Aims

**PyAutoGalaxy** has three core aims:

- **Big Data**: Scaling automated Sérsic fitting to extremely large datasets, *accelerated with JAX on GPUs and using tools like an SQL database to **build a scalable scientific workflow***.
- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse Fitting, Irregular Meshes) that go beyond just simple Sérsic fitting.
- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be fitted independently or simultaneously.

A complete overview of the software's aims is provided in our [Journal of Open Source Software paper](https://joss.theoj.org/papers/10.21105/joss.04475).

# Community & Contributing

**PyAutoGalaxy** is built in the open by its users: everyone is welcome to ask questions,
share what they have made with it, and contribute.

Questions, ideas and bug reports: the [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions).
Chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

Community-built tools and tutorials, and how to contribute: the [**PyAutoGalaxy** community page](https://pyautogalaxy.readthedocs.io/en/latest/general/community.html).

# HowToGalaxy

For users less familiar with galaxy analysis, Bayesian inference, and scientific analysis, you may wish to read through
the **HowToGalaxy** lectures. These introduce the basic principles of galaxy modeling and Bayesian inference, with
the material pitched at undergraduate level and above.

A complete overview of the lectures [is provided on the HowToGalaxy readthedocs page](https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html), and the notebooks themselves live in the [PyAutoLabs/HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy) repository.

# Citations

Information on how to cite **PyAutoGalaxy** in publications can be found [on the citations page](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/CITATIONS.md).

```{toctree}
:caption: 'Overview:'
:hidden: true
:maxdepth: 1

overview/overview_1_start_here
overview/overview_2_new_user_guide
overview/overview_3_features
```

```{toctree}
:caption: 'Installation:'
:hidden: true
:maxdepth: 1

installation/overview
installation/conda
installation/pip
installation/numba
installation/source
installation/troubleshooting
```

```{toctree}
:caption: 'General:'
:hidden: true
:maxdepth: 1

general/workspace
general/configs
general/model_cookbook
general/likelihood_function
general/citations
general/credits
general/community
```

```{toctree}
:caption: 'Tutorials:'
:hidden: true
:maxdepth: 1

howtogalaxy/howtogalaxy
howtogalaxy/chapter_1_introduction
howtogalaxy/chapter_2_modeling
howtogalaxy/chapter_3_pixelizations
howtogalaxy/chapter_4_scaling_up_galaxies
howtogalaxy/chapter_optional
```

```{toctree}
:caption: 'API Reference:'
:hidden: true
:maxdepth: 1

api/data
api/light
api/galaxy
api/mass_budget
api/fitting
api/modeling
api/pixelization
api/plot
api/source
```
