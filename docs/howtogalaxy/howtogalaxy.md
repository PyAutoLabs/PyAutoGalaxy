(howtogalaxy)=

# HowToGalaxy Lectures

The best way to learn **PyAutoGalaxy** is by going through the **HowToGalaxy** lecture series, which lives in its own
repository at [PyAutoLabs/HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy).

The lectures are provided as Jupyter notebooks (and Python scripts), and they are linked to via this readthedocs. The
lectures are composed of five chapters

- **Introduction** - An introduction to galaxy morphology and structure using **PyAutoGalaxy**.
- **Lens Modeling** - How to model galaxies, including a primer on Bayesian non-linear analysis.
- **Search Chaining** - How to fit complex lens models using non-linear search chaining.
- **Pixelizations** - How to perform pixelized reconstructions of a galaxy.

## How to Tackle HowToGalaxy

The **HowToGalaxy** lecture series current sits at 4 chapters, and each will take a day or so to go through
properly. You probably want to be modeling galaxies faster than that! Furthermore, the concepts
in the later chapters are pretty challenging, and familiarity and modeling is desirable before
you tackle them.

Therefore, we recommend that you complete chapters 1 & 2 and then apply what you've learnt to the modeling of simulated
and real galaxy data, using the scripts found in the
[autogalaxy_workspace](https://github.com/PyAutoLabs/autogalaxy_workspace). Once you're happy
with the results and confident with your use of **PyAutoGalaxy**, you can then begin to cover the advanced functionality
covered in chapters 3 & 4.

## Jupyter Notebooks

The tutorials are supplied as Jupyter Notebooks, which come with a '.ipynb' suffix. For those new to Python, Jupyter
Notebooks are a different way to write, view and use Python code. Compared to the traditional Python scripts, they allow:

- Small blocks of code to be viewed and run at a time
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the HowToFit lecture series, therefore I recommend you get yourself
a Jupyter notebook viewer (<https://jupyter.org/>) if you havent done so already.

If you *really* want to use Python scripts, all tutorials are also supplied as `.py` files in the `scripts` folder
of the [HowToGalaxy repository](https://github.com/PyAutoLabs/HowToGalaxy).

## Visualization

Before beginning the **HowToGalaxy** lecture series, in chapter 1 you should do 'tutorial_0_visualization'. This will
take you through how **PyAutoGalaxy** interfaces with matplotlib to perform visualization and will get you setup such
that images and figures display correctly in your Jupyter notebooks.

## Code Style and Formatting

You may notice the style and formatting of our Python code looks different to what you are used to. For example, it
is common for brackets to be placed on their own line at the end of function calls, the inputs of a function or
class may be listed over many separate lines and the code in general takes up a lot more space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact, lots of people do,
which is why we use an auto-formatter to produce the code in a standardized format. If you're interested in the style
and would like to adapt it to your own code, check out the Python auto-code formatter 'black'.

<https://github.com/python/black>
