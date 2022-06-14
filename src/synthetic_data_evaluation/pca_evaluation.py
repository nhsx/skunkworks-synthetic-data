import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_preparation.evaluation_data_prep import save_image_as_bytes


class pca_evaluation:
    def __init__(self, standardscaler: bool = True, **kwargs: dict):
        self.standardscaler = standardscaler
        self.pca = self.define_pca(**kwargs)

    def define_pca(self, **kwargs) -> PCA:
        """Pass parameters to PCA. More details on the paramaters availible:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"""
        params_dict = {}
        for param in PCA()._get_param_names():
            if param in kwargs:
                params_dict[param] = kwargs[param]
        return PCA(**params_dict)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method runs the PCA based on settings specified.
        More details here:
         https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        inputs:
        - data - the dataset you that will be used in the analysis

        returns:
        - principal components in a dataframe

        """
        if self.standardscaler:
            data = StandardScaler().fit_transform(data)

        principalComponents = self.pca.fit_transform(data)
        return pd.DataFrame(principalComponents)

    def run_comparison(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        show_plot: bool = False,
    ) -> str:

        """This method uses the calculate method to run the PCA for the
        synthetic and real data.

        inputs:
        - real data
        - synthetic data
        - show plot - will display plot when True selected.

        returns:
        - save_image_as_bytes() - the plot is returned as a "utf-8" string so it can be stored in the kedro
        pipline and appear in the html report.
        """
        # Run PCA
        syn_data_pca = self.calculate(synthetic_data)
        real_data_pca = self.calculate(real_data)

        # Plot PCA outputs
        fig, ax = plt.subplots(figsize=(10, 10))
        syn_data_pca.plot.scatter(ax=ax, x=0, y=1, c="r", label="synthetic_data")
        real_data_pca.plot.scatter(ax=ax, x=0, y=1, c="b", label="real_data")
        ax.set_title("PCA Real Data and Synthetic data", fontsize=20)
        ax.grid()
        plt.legend()
        # Output plot if True
        if show_plot:
            plt.show()

        # Return plot as "utf-8" string
        return save_image_as_bytes()
