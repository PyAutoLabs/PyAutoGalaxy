import autofit as af

from autoarray import exc

from autogalaxy.ellipse.model.plotter import PlotterEllipse


class VisualizerEllipse(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        plotter = PlotterEllipse(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        plotter.imaging(dataset=analysis.dataset)

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit ellipses over the data, superimposed with contours showing the traced path of the
          ellipse around the data and thus how well it fits the data.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        """
        fit_list = analysis.fit_list_from(instance=instance)

        plotter = PlotterEllipse(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )
        plotter.imaging(dataset=analysis.dataset)

        plotter.fit_ellipse(fit_list=fit_list)
